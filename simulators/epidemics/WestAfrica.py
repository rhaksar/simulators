from collections import defaultdict
import numpy as np
import pickle
import pkgutil

from simulators.epidemics.RegionElements import Region
from simulators.Simulator import Simulator


class WestAfrica(Simulator):
    """
    A simulator for the 2014 Ebola outbreak in West Africa.
    """
    def __init__(self, initial_outbreak, rng=None,
                 eta=None, region_model='exponential'):
        """
        Initializes a simulation object. Each element is a Region.

        :param initial_outbreak: dictionary describing the Regions that are initially infected.
                                 Each key should return a count of how long the Region has been infected.
        :param rng: random number generator seed for deterministic sampling
        :param eta: disease propagation parameter, as a dictionary with Region name as keys
        :param region_model: simulation model for Region elements, either 'linear' or 'exponential'
        """
        Simulator.__init__(self)

        if region_model == 'linear':
            self.eta = defaultdict(lambda: 0.17) if eta is None else eta
        elif region_model == 'exponential':
            self.eta = defaultdict(lambda: 0.08) if eta is None else eta

        # load dictionary describing the connections between Regions. Keys are region names (string) and refer to a
        # dictionary:
        #      name (string) : {'edges': list of names (strings) indicating Region connections,
        #                       'pos': tuple, for visualization}
        data = pkgutil.get_data('simulators', 'epidemics/west_africa_graph.pkl')
        graph = pickle.loads(data)

        self.dims = len(graph.keys())
        self.initial_outbreak = initial_outbreak

        # create a collection of Regions based on provided graph
        self.group = dict()
        self.counter = dict()  # a count of how long each Region has been in the infected state
        for idx, name in enumerate(graph.keys()):
            self.group[name] = Region(self.eta[name], name=name, position=graph[name]['pos'],
                                      numeric_id=idx, model=region_model)
            self.group[name].neighbors = graph[name]['edges']
            self.counter[name] = 0

            # set initial outbreak
            if name in self.initial_outbreak.keys():
                self.group[name].set_infected()
                self.counter[name] = self.initial_outbreak[name]

        self.rng = rng
        if rng is not None:
            np.random.seed(rng)

        self.iter = 0
        self.end = False
        return

    def reset(self):
        """
        Reset simulation object to initialization.
        """
        # reset Regions and counter
        for name in self.group.keys():
            self.group[name].reset()
            self.counter[name] = 0

            # reset to initial outbreak
            if name in self.initial_outbreak.keys():
                self.group[name].set_infected()
                self.counter[name] = self.initial_outbreak[name]

        if self.rng is not None:
            np.random.seed(self.rng)

        self.iter = 0
        self.end = False
        return

    def dense_state(self):
        """
        Create a representation of the state of each Region.

        :return: a dictionary where the Region name refers to its state
        """
        return {name: self.group[name].state for name in self.group.keys()}

    def update(self, control=None):
        """
        Update the simulator one time step.

        :param control: collection to map Region name to control for each Region,
                        which is a tuple of (delta_eta, delta_nu)
        """
        if self.end:
            print('process has terminated')

        if control is None:
            control = defaultdict(lambda: (0, 0))

        # determine next state for each Region
        for name in self.group.keys():
            self.group[name].next(self.group, control[name])

        # assume simulation will end this time step
        self.end = True
        for name in self.group.keys():
            # check if there are any infected Regions
            if self.group[name].is_infected(self.group[name].next_state):
                self.end = False
                self.counter[name] += 1

            # apply next state to all elements
            self.group[name].update()

        self.iter += 1
        return
