import itertools
from collections import defaultdict
import numpy as np

from fires.ForestElements import Tree
from Simulator import Simulator


class LatticeForest(Simulator):
    """
    A simulator for a forest fire using a discrete probabilistic lattice model.
    """
    def __init__(self, dimension, rng=None, initial_fire=None,
                 alpha=None, beta=None, tree_model='exponential'):
        """
        Initializes a simulation object. Each element is a Tree with a (row, col) position.

        :param dimension: size of forest, integer or (height, width)
                          if an integer, the forest is square
        :param rng: random number generator seed for deterministic sampling
        :param initial_fire: collection of (row, col) coordinates describing positions of initial fires
        :param alpha: fire propagation parameter, as a dictionary with (row, col) as keys
        :param beta: fire persistence parameter, as a dictionary with (row, col) as keys)
        :param tree_model: simulation model for Tree elements, either 'linear' or 'exponential'
        """
        Simulator.__init__(self)

        self.dims = (dimension, dimension) if isinstance(dimension, int) else dimension
        if tree_model == 'exponential':
            self.alpha = defaultdict(lambda: 0.2763) if alpha is None else alpha
        elif tree_model == 'linear':
            self.alpha = defaultdict(lambda: 0.2) if alpha is None else alpha
        self.beta = defaultdict(lambda: np.exp(-1/10)) if beta is None else beta

        # statistics for the simulation: number of [healthy, fire, burnt] trees
        self.stats = np.zeros(3).astype(np.uint32)
        self.stats[0] += self.dims[0]*self.dims[1]

        # deterministic sampling
        self.rng = rng
        if self.rng is not None:
            np.random.seed(self.rng)

        # the forest is a group of Trees
        self.group = dict()
        for r in range(self.dims[0]):
            for c in range(self.dims[1]):
                self.group[(r, c)] = Tree(self.alpha[(r, c)], self.beta[(r, c)], position=np.array([r, c]),
                                          numeric_id=r*self.dims[1]+c, model=tree_model)

                # neighbors are adjacent Trees on the lattice
                if 0 <= r+1 < self.dims[0]:
                    self.group[(r, c)].neighbors.append((r+1, c))
                if 0 <= r-1 < self.dims[0]:
                    self.group[(r, c)].neighbors.append((r-1, c))
                if 0 <= c+1 < self.dims[1]:
                    self.group[(r, c)].neighbors.append((r, c+1))
                if 0 <= c-1 < self.dims[1]:
                    self.group[(r, c)].neighbors.append((r, c-1))

        # start initial fire
        self.iter = 0
        self.fires = []  # list containing (row, col) positions corresponding to Trees on fire
        self.initial_fire = initial_fire
        self._start_fire()

        self.end = False
        self.early_end = False
        return

    def _start_fire(self):
        """
        Helper method to specify initial fire locations in the forest.
        """
        # apply initial condition if specified
        if self.initial_fire is not None:
            self.fires = self.initial_fire
            for p in self.initial_fire:
                self.group[p].set_on_fire()

            self.stats[0] -= len(self.initial_fire)
            self.stats[1] += len(self.initial_fire)

            return

        # start a 4x4 square of fires at center
        # if forest size is too small, start a single fire at the center
        r_center = np.floor((self.dims[0]-1)/2).astype(np.uint8)
        c_center = np.floor((self.dims[1]-1)/2).astype(np.uint8)

        delta_r = [0] if self.dims[0]<4 else [k for k in range(-1, 3)]
        delta_c = [0] if self.dims[1]<4 else [k for k in range(-1, 3)]
        deltas = itertools.product(delta_r, delta_c)

        for (dr, dc) in deltas:
            r, c = r_center+dr, c_center+dc
            self.fires.append((r, c))
            self.group[(r, c)].set_on_fire()

        self.stats[0] -= len(self.fires)
        self.stats[1] += len(self.fires)
        return

    def reset(self):
        """
        Reset the simulation object to its initial configuration.
        """
        # reset statistics
        self.stats = np.zeros(3).astype(np.uint32)
        self.stats[0] += self.dims[0]*self.dims[1]

        # reset elements
        for element in self.group.values():
            element.reset()

        # reset to initial condition
        self.iter = 0
        self.fires = []
        self._start_fire()
        if self.rng is not None:
            np.random.seed(self.rng)

        self.end = False
        self.early_end = False
        return

    def dense_state(self):
        """
        Creates a representation of the state of each Tree.

        :return: 2D numpy array where each position (row, col) corresponds to a Tree state
        """
        return np.array([[self.group[(r, c)].state for c in range(self.dims[1])]
                         for r in range(self.dims[0])])

    def update(self, control=None):
        """
        Update the simulator one time step.

        :param control: collection to map (row, col) to control for each Tree,
                        which is a tuple of (delta_alpha, delta_beta)
        """
        if self.end:
            print("fire extinguished")
            return

        if control is None:
            control = defaultdict(lambda: (0, 0))

        # assume that the fire cannot spread further this step,
        # which occurs when no healthy Trees have a neighbor that is on fire
        self.early_end = True

        # list of (row, col) positions corresponding to Trees caught on fire this time step
        add = []
        # list of (row, col) positions corresponding to healthy Trees that have been sampled to determine
        # if they will catch on fire
        checked = []

        # fire spreading check:
        #   iterate over current fires, find their neighbors that are healthy, and sample
        #   to determine if the healthy Tree catches on fire
        # all other Tree states will not change
        for f in self.fires:
            for fn in self.group[f].neighbors:
                if fn not in checked and self.group[fn].is_healthy(self.group[fn].state):

                    self.early_end = False

                    # calculate next state
                    self.group[fn].next(self.group, control[fn])
                    if self.group[fn].is_on_fire(self.group[fn].next_state):
                        add.append(fn)

                    checked.append(fn)

            # determine if the current Tree on fire will extinguish this time step
            self.group[f].next(self.group, control[f])
            if self.group[f].is_burnt(self.group[f].next_state):
                self.stats[1] -= 1
                self.stats[2] += 1

        # apply next state to all elements
        for element in self.group.values():
            element.update()

        # retain Trees that are still on fire
        self.fires = [f for f in self.fires
                      if self.group[f].is_on_fire(self.group[f].state)]

        # add Trees that caught on fire
        self.fires.extend(add)
        self.stats[0] -= len(add)
        self.stats[1] += len(add)

        self.iter += 1

        if not self.fires:
            self.early_end = True
            self.end = True
            return

        return
