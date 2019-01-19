import itertools
from collections import defaultdict
import numpy as np

from fires.ForestElements import Tree


class LatticeForest(object):

    def __init__(self, dimension, rng=None, initial_fire=None,
                 alpha=None, beta=None, tree_model='exponential'):

        self.dims = (dimension, dimension) if isinstance(dimension, int) else dimension

        self.alpha = defaultdict(lambda: 0.2763) if alpha is None else alpha
        self.beta = defaultdict(lambda: np.exp(-1/10)) if beta is None else beta

        self.stats = np.zeros(3).astype(np.uint32)
        self.stats[0] += self.dims[0]*self.dims[1]

        if rng is not None:
            np.random.seed(rng)

        self.forest = dict()
        for r in range(self.dims[0]):
            for c in range(self.dims[1]):
                self.forest[(r, c)] = Tree(self.alpha[(r, c)], self.beta[(r, c)],
                                           position=np.array([r, c]), model=tree_model)

                if 0 <= r+1 < self.dims[0]:
                    self.forest[(r, c)].neighbors.append((r+1, c))
                if 0 <= r-1 < self.dims[0]:
                    self.forest[(r, c)].neighbors.append((r-1, c))
                if 0 <= c+1 < self.dims[1]:
                    self.forest[(r, c)].neighbors.append((r, c+1))
                if 0 <= c-1 < self.dims[1]:
                    self.forest[(r, c)].neighbors.append((r, c-1))

        self.iter = 0
        self.fires = []
        self.initial_fire = initial_fire
        self._start_fire(self.initial_fire)

        self.end = False
        self.early_end = False
        return

    def _start_fire(self, initial_fire):
        """
        Helper method to specify initial fire locations in the forest.

        Inputs:
         initial_fire:

        Outputs:
         None
        """

        if initial_fire is not None:
            self.fires = initial_fire
            for p in initial_fire:
                self.forest[p].set_on_fire()

            self.stats[0] -= len(initial_fire)
            self.stats[1] += len(initial_fire)

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
            self.forest[(r, c)].set_on_fire()

        self.stats[0] -= len(self.fires)
        self.stats[1] += len(self.fires)
        return

    def reset(self):
        """
        Method to reset the simulation object to its initial configuration.

        Inputs/Outputs:
         None
        """
        self.stats = np.zeros(3).astype(np.uint32)
        self.stats[0] += self.dims[0]*self.dims[1]

        for element in self.forest.values():
            element.reset()

        self._start_fire(self.initial_fire)

        self.end = False
        self.early_end = False
        return

    def dense_state(self):
        return np.array([[self.forest[(r, c)].state for c in range(self.dims[1])]
                         for r in range(self.dims[0])])

    def update(self, control):
        if self.end:
            print("fire extinguished")
            return

        self.early_end = True

        add = []
        checked = []

        for f in self.fires:
            for fn in self.forest[f].neighbors:
                if fn not in checked and self.forest[fn].is_healthy(self.forest[fn].state):

                    self.early_end = False

                    self.forest[fn].next(self.forest, control[fn])
                    if self.forest[fn].is_on_fire(self.forest[fn].next_state):
                        add.append(fn)

                    checked.append(fn)

            self.forest[f].next(self.forest, control[f])
            if self.forest[f].is_burnt(self.forest[f].next_state):
                self.stats[1] -= 1
                self.stats[2] += 1

        for element in self.forest.values():
            element.update()

        self.fires = [f for f in self.fires
                      if self.forest[f].is_on_fire(self.forest[f].state)]

        self.fires.extend(add)
        self.stats[0] -= len(add)
        self.stats[1] += len(add)

        self.iter += 1

        if not self.fires:
            self.early_end = True
            self.end = True
            return

        return
