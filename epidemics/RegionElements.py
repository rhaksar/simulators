import numpy as np

from Element import Element


class Region(Element):
    def __init__(self, eta, name=None, position=None, numeric_id=None,
                 model='exponential'):
        Element.__init__(self)

        self.healthy = 0
        self.infected = 1
        self.immune = 2

        self.state = self.healthy
        self.next_state = self.state
        self.state_space = [self.healthy,
                            self.infected,
                            self.immune]

        self.numeric_id = numeric_id
        self.name = name
        self.position = position

        self.model = model
        self.eta = eta

        self.neighbors = []
        self.neighbors_states = []
        self.neighbors_types = [Region]

    def reset(self):
        self.state = self.healthy
        self.next_state = self.state
        return

    def update(self):
        self.state = self.next_state
        return

    def next(self, group, control):
        self.next_state = self.state
        if self.state != self.immune:
            number_neighbors_on_fire = None
            if self.state == self.healthy:
                self.query_neighbors(group)
                number_neighbors_on_fire = self.neighbors_states.count(True)

            transition_p = self.dynamics((self.state, number_neighbors_on_fire, self.state+1), control)
            if np.random.rand() < transition_p:
                self.next_state = self.state + 1

    def query_neighbors(self, group):
        self.neighbors_states = [group[j].is_infected(group[j].state) for j in self.neighbors
                                 if any(isinstance(group[j], t) for t in self.neighbors_types)]

    def dynamics(self, state_and_next_state, control):
        if self.model == 'exponential':
            return self.dynamics_exponential(state_and_next_state, control)

    def dynamics_exponential(self, state_and_next_state, control):
        state, number_infected_neighbors, next_state = state_and_next_state
        delta_eta, delta_nu = control

        if state == self.healthy:
            if next_state is self.healthy:
                return (1 - self.eta + delta_eta)**number_infected_neighbors
            elif next_state == self.infected:
                return 1 - (1 - self.eta + delta_eta)**number_infected_neighbors
            else:
                return 0

        elif state == self.infected:
            if next_state == self.healthy:
                return 0
            elif next_state == self.infected:
                return 1 - delta_nu
            else:
                return delta_nu

        else:
            if next_state == self.immune:
                return 1
            else:
                return 0

    def is_healthy(self, query):
        return query == self.healthy

    def is_infected(self, query):
        return query == self.infected

    def is_immune(self, query):
        return query == self.immune

    def set_infected(self):
        self.state = self.infected
