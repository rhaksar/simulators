from collections import defaultdict
import itertools
import numpy as np

from simulators.fires.ForestElements import Tree, SimpleUrban
from simulators.Simulator import Simulator


class UrbanForest(Simulator):
    """
    A simulator for a lattice-based forest with urban elements. Based on the LatticeForest simulator.
    """

    def __init__(self, dimension, urban_width, rng=None, initial_fire=None,
                 alpha=None, beta=None, tree_model='exponential'):

        # LatticeForest.__init__(self, dimension, rng=rng, initial_fire=initial_fire,
        #                        alpha=alpha, beta=beta, tree_model=tree_model)
        Simulator.__init__(self)

        self.dims = (dimension, dimension) if isinstance(dimension, int) else dimension
        if tree_model == 'exponential':
            self.alpha = defaultdict(lambda: 0.2763) if alpha is None else alpha
        elif tree_model == 'linear':
            self.alpha = defaultdict(lambda: 0.2) if alpha is None else alpha
        self.beta = defaultdict(lambda: np.exp(-1/10)) if beta is None else beta

        self.rng = rng
        self.random_state = np.random.RandomState(self.rng)

        self.urban = []
        self.urban_width = urban_width

        # the forest is a group of Trees and SimpleUrban elements
        self.group = dict()
        for r in range(self.dims[0]):
            for c in range(self.dims[1]):

                # urban elements compose the right-most edge of the lattice
                if c >= self.dims[1]-self.urban_width:
                    self.group[(r, c)] = SimpleUrban(self.alpha[(r, c)], self.beta[(r, c)], position=np.array([r, c]),
                                                     numeric_id=r*self.dims[1]+c)
                    self.urban.append((r, c))

                # all other elements are trees
                else:
                    self.group[(r, c)] = Tree(self.alpha[(r, c)], self.beta[(r, c)], position=np.array([r, c]),
                                              numeric_id=r*self.dims[1]+c, model=tree_model)

                if 0 <= r+1 < self.dims[0]:
                    self.group[(r, c)].neighbors.append((r+1, c))
                if 0 <= r-1 < self.dims[0]:
                    self.group[(r, c)].neighbors.append((r-1, c))
                if 0 <= c+1 < self.dims[1]:
                    self.group[(r, c)].neighbors.append((r, c+1))
                if 0 <= c-1 < self.dims[1]:
                    self.group[(r, c)].neighbors.append((r, c-1))

        self.stats_trees = np.zeros(3).astype(np.int)
        self.stats_trees[0] += self.dims[0]*self.dims[1] - len(self.urban)

        self.stats_urban = np.zeros(4).astype(np.int)
        self.stats_urban[0] += len(self.urban)

        # start initial fire
        self.iter = 0
        self.fires = []
        self.initial_fire = initial_fire
        self._start_fire()

        self.early_end = False
        self.end = False

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

                if isinstance(self.group[p], Tree):
                    self.stats_trees[0] -= 1
                    self.stats_trees[1] += 1
                elif isinstance(self.group[p], SimpleUrban):
                    self.stats_urban[0] -= 1
                    self.stats_urban[1] += 1

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

            if isinstance(self.group[(r, c)], Tree):
                self.stats_trees[0] -= 1
                self.stats_trees[1] += 1
            elif isinstance(self.group[(r, c)], SimpleUrban):
                self.stats_urban[0] -= 1
                self.stats_urban[1] += 1

        return

    def reset(self):
        """
        Reset the simulation object to its initial configuration.
        """
        # reset statistics
        self.stats_trees = np.zeros(3).astype(np.int)
        self.stats_trees[0] += self.dims[0]*self.dims[1] - len(self.urban)

        self.stats_urban = np.zeros(4).astype(np.int)
        self.stats_urban[0] += len(self.urban)

        # reset elements
        for element in self.group.values():
            element.reset()

        # reset to initial condition
        self.iter = 0
        self.fires = []
        self._start_fire()
        self.random_state = np.random.RandomState(self.rng)

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

        :param control: collection to map (row, col) to control for each Element,
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

        # list of (row, col) positions corresponding to elements caught on fire this time step
        add = []
        # list of (row, col) positions corresponding to healthy elements that have been sampled to determine
        # if they will catch on fire
        checked = []

        # calculate next state for urban elements not on fire, in case they are removed from the lattice
        do_not_check = []
        for u in self.urban:
            if self.group[u].is_healthy(self.group[u].state):
                self.group[u].next(self.group, control[u], self.random_state)

                if self.group[u].is_removed(self.group[u].next_state):
                    self.stats_urban[0] -= 1
                    self.stats_urban[3] += 1
                    do_not_check.append(u)

        # fire spreading check:
        #   iterate over current fires, find their neighbors that are healthy, and sample
        #   to determine if the healthy element catches on fire
        for f in self.fires:
            for fn in self.group[f].neighbors:
                if fn not in checked and self.group[fn].is_healthy(self.group[fn].state):

                    if isinstance(self.group[fn], SimpleUrban) and fn in do_not_check:
                        continue

                    self.early_end = False

                    # calculate next state
                    self.group[fn].next(self.group, control[fn], self.random_state)
                    if self.group[fn].is_on_fire(self.group[fn].next_state):
                        add.append(fn)

                    checked.append(fn)

            # determine if the current element on fire will extinguish this time step
            self.group[f].next(self.group, control[f], self.random_state)
            if self.group[f].is_burnt(self.group[f].next_state):
                if isinstance(self.group[f], Tree):
                    self.stats_trees[1] -= 1
                    self.stats_trees[2] += 1
                elif isinstance(self.group[f], SimpleUrban):
                    self.stats_urban[1] -= 1
                    self.stats_urban[2] += 1

        # apply next state to all elements
        for element in self.group.values():
            element.update()

        # retain elements that are still on fire
        self.fires = [f for f in self.fires if self.group[f].is_on_fire(self.group[f].state)]

        # add elements that caught on fire
        self.fires.extend(add)
        for a in add:
            if isinstance(self.group[a], Tree):
                self.stats_trees[0] -= 1
                self.stats_trees[1] += 1
            elif isinstance(self.group[a], SimpleUrban):
                self.stats_urban[0] -= 1
                self.stats_urban[1] += 1

        self.iter += 1

        if not self.fires:
            self.early_end = True
            self.end = True
            return

        return
