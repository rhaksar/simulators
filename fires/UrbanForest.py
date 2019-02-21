from collections import defaultdict
import numpy as np

from fires.ForestElements import Tree, SimpleUrban
from fires.LatticeForest import LatticeForest


class UrbanForest(LatticeForest):
    """
    A simulator for a lattice-based forest with urban elements. Based on the LatticeForest simulator.
    """

    def __init__(self, dimension, urban_width, rng=None, initial_fire=None,
                 alpha=None, beta=None, tree_model='exponential'):

        LatticeForest.__init__(self, dimension, rng=rng, initial_fire=initial_fire,
                               alpha=alpha, beta=beta, tree_model=tree_model)

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

                if 0 <= r + 1 < self.dims[0]:
                    self.group[(r, c)].neighbors.append((r + 1, c))
                if 0 <= r - 1 < self.dims[0]:
                    self.group[(r, c)].neighbors.append((r - 1, c))
                if 0 <= c + 1 < self.dims[1]:
                    self.group[(r, c)].neighbors.append((r, c + 1))
                if 0 <= c - 1 < self.dims[1]:
                    self.group[(r, c)].neighbors.append((r, c - 1))

        # start initial fire
        self.fires = []
        self._start_fire()

        self.early_end = False
        self.end = False

        return

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

        # calculate next state for urban elements not on fire, in case they are removed from the lattice
        for u in self.urban:
            if u not in self.fires:
                self.group[u].next(self.group, control[u])

        # apply next state to all elements
        for element in self.group.values():
            element.update()

        # retain Trees that are still on fire
        self.fires = [f for f in self.fires if self.group[f].is_on_fire(self.group[f].state)]

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
