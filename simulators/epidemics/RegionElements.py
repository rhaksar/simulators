import numpy as np

from simulators.Element import Element


class Region(Element):
    """
    Implementation of a simple region for simulating a disease epidemic.
    """
    def __init__(self, eta, name=None, position=None, numeric_id=None,
                 model='exponential'):
        Element.__init__(self)

        # states and state space definition
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

        # region model and disease spread parameter
        self.model = model
        self.eta = eta

        # data structures for information from neighbors
        self.neighbors = []
        self.neighbors_states = []
        self.neighbors_types = [Region]

    def reset(self):
        """
        Reset the Region to initialization.
        """
        self.state = self.healthy
        self.next_state = self.state
        return

    def update(self):
        """
        Set the state to the sampled next state.
        'next' should be called prior to 'update'
        :return:
        """
        self.state = self.next_state
        return

    def next(self, group, control=(0, 0)):
        """
        Sample and set the next state. Simplifies implementation of a Markov process.
        """
        # assume state does not change
        self.next_state = self.state

        if self.state != self.immune:
            # only healthy state needs information from neighbors
            number_neighbors_on_fire = None
            if self.state == self.healthy:
                self.neighbors_states = self.query_neighbors(group)
                number_neighbors_on_fire = self.neighbors_states.count(True)

            # calculate transition probability and sample
            transition_p = self.dynamics((self.state, number_neighbors_on_fire, self.state+1), control)
            if np.random.rand() < transition_p:
                self.next_state = self.state + 1

    def query_neighbors(self, group):
        """
        Determine whether or not neighbors are infected.
        Supported neighbor types are defined by self.neighbors_types
        """
        return [group[j].is_infected(group[j].state) for j in self.neighbors
                if any(isinstance(group[j], t) for t in self.neighbors_types)]

    def dynamics(self, state_and_next_state, control=(0, 0)):
        """
        Calculate a transition probability:
            state - healthy/infected/immune, number of infected neighbors
            next_state - healthy/infected/immune
            control - (delta_eta, delta_nu)
        """
        if self.model == 'linear':
            return self.dynamics_linear(state_and_next_state, control)
        elif self.model == 'exponential':
            return self.dynamics_exponential(state_and_next_state, control)

    def dynamics_linear(self, state_and_next_state, control=(0, 0)):
        state, number_infected_neighbors, next_state = state_and_next_state
        delta_eta, delta_nu = control

        if state == self.healthy:
            if next_state == self.healthy:
                return 1 - (self.eta - delta_eta)*number_infected_neighbors
            elif next_state == self.infected:
                return (self.eta - delta_eta)*number_infected_neighbors
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

    def dynamics_exponential(self, state_and_next_state, control=(0, 0)):
        """
        Implementation of transition distribution. The transition from healthy to infected
        is an exponential function of the number of infected neighbors.

        Note that the process is modeled as non-self-terminating! Infected Regions will never transition
        to immune unless control effort (delta_nu > 0) is applied!
        """
        state, number_infected_neighbors, next_state = state_and_next_state
        delta_eta, delta_nu = control

        if state == self.healthy:
            if next_state == self.healthy:
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

    # helper methods for querying/setting states
    def is_healthy(self, query):
        return query == self.healthy

    def is_infected(self, query):
        return query == self.infected

    def is_immune(self, query):
        return query == self.immune

    def set_infected(self):
        self.state = self.infected
