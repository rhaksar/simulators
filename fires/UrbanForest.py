import numpy as np

from fires.ForestElements import Tree, SimpleUrban
from fires.LatticeForest import LatticeForest


class UrbanForest(LatticeForest):

    def __init__(self, dimension, urban_width=10, rng=None, initial_fire=None,
                 alpha=None, beta=None, tree_model='exponential'):

        LatticeForest.__init__(self, dimension, rng=rng, initial_fire=initial_fire,
                               alpha=alpha, beta=beta, tree_model=tree_model)

        self.urban_elements = []

        # the forest is a group of Trees and SimpleUrban elements
        self.group = dict()
        for r in range(self.dims[0]):
            for c in range(self.dims[1]):

                if c >= self.dims[1]-urban_width:
                    self.group[(r, c)] = SimpleUrban(self.alpha[(r, c)], self.beta[(r, c)], position=np.array([r, c]),
                                                     numeric_id=r*self.dims[1]+c)
                    self.urban_elements.append((r, c))

                else:
                    self.group[(r, c)] = Tree(self.alpha[(r, c)], self.beta[(r, c)], position=np.array([r, c]),
                                              numeric_id=r*self.dims[1]+c, model=tree_model)

                if 0 <= r + 1 < self.dims[0]:
                    self.group[(r, c)].neighbors.append((r + 1, c))
                if 0 <= r - 1 < self.dims[0]:
                    self.group[(r, c)].neighbors.append((r - 1, c))
                if 0 <= c + 1 < self.dims[1]:
                    self.group[(r, c)].neighbors.append((r, c + 1))
                if 0 <= c - 1 < self.dims[1]:
                    self.group[(r, c)].neighbors.append((r, c - 1))

        # start initial fire
        self._start_fire()

        return
