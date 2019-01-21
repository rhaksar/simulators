from collections import defaultdict
import numpy as np

from epidemics.RegionElements import Region


class WestAfrica(object):
    def __init__(self, graph, initial_outbreak, rng=None,
                 eta=None, region_model='exponential'):
        if region_model == 'exponential':
            self.eta = defaultdict(lambda: 0.08) if eta is None else eta
        elif region_model == 'linear':
            self.eta = defaultdict(lambda: 0.17) if eta is None else eta

        self.dims = len(graph.keys())
        self.initial_outbreak = initial_outbreak

        self.group = dict()
        self.counter = dict()
        for name in graph.keys():
            self.group[name] = Region(self.eta[name], name=name,
                                      position=graph[name]['pos'], model=region_model)
            self.group[name].neighbors = graph[name]['edges']
            self.counter[name] = 0

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
        for name in self.group.keys():
            self.group[name].reset()
            self.counter[name] = 0

            if name in self.initial_outbreak.keys():
                self.group[name].set_infected()
                self.counter[name] = self.initial_outbreak[name]

        if self.rng is not None:
            np.random.seed(self.rng)

        self.iter = 0
        self.end = False
        return

    def dense_state(self):
        return {name: self.group[name].state for name in self.group.keys()}

    def update(self, control):
        if self.end:
            print('process has terminated')

        for name in self.group.keys():
            self.group[name].next(self.group, control[name])

        self.end = True
        for name in self.group.keys():
            if self.group[name].is_infected(self.group[name].next_state):
                self.end = False
                self.counter[name] += 1

            self.group[name].update()

        self.iter += 1
        return
