import numpy as np
import sys
import warnings

from Element import Element


class Tree(Element):
    """
    Implmentation of a simple Tree simulation object for simulating a forest fire.
    """
    def __init__(self, alpha, beta, position=None, numeric_id=None,
                 model='exponential'):
        Element.__init__(self)

        # states and state space definition
        self.healthy = 0
        self.on_fire = 1
        self.burnt = 2

        self.state = self.healthy
        self.next_state = self.state
        self.state_space = [self.healthy,
                            self.on_fire,
                            self.burnt]

        # position can be used for determining neighbors
        self.position = position
        self.numeric_id = numeric_id

        # model type and parameters
        self.model = model
        self.alpha = alpha
        self.beta = beta

        # data structures for information from neighbors
        self.neighbors = []
        self.neighbors_states = []
        self.neighbors_types = [Tree]
        return

    def reset(self):
        """
        Reset the Tree to initialization.
        """
        self.state = self.healthy
        self.next_state = self.state
        return

    def update(self):
        """
        Set the state to the calculated next state.
        The method 'next' should be called first.
        """
        self.state = self.next_state
        return

    def next(self, forest, control=(0, 0)):
        """
        Sample, but don't apply, the next state.
        This makes implementation of a Markov process simpler.
        """
        # first assume the state will not change
        self.next_state = self.state

        if self.state != self.burnt:
            # Only the healthy state needs to know information from the neighbors
            number_neighbors_on_fire = None
            if self.state == self.healthy:
                self.neighbors_states = self.query_neighbors(forest)
                number_neighbors_on_fire = self.neighbors_states.count(True)

            # calculate transition probability and sample
            transition_p = self.dynamics((self.state, number_neighbors_on_fire, self.state+1), control)
            if np.random.rand() < transition_p:
                self.next_state = self.state + 1
        return

    def query_neighbors(self, forest):
        """
        Determine how many neighboring Elements are on fire.
        Supported neighbor types are defined by self.neighbors_types.
        """
        return [forest[j].is_on_fire(forest[j].state) for j in self.neighbors
                if any(isinstance(forest[j], t) for t in self.neighbors_types)]

    def dynamics(self, state_and_next_state, control=(0, 0)):
        """
        Calculate a transition probability:
            state - healthy/on_fire/burnt, number of neighbors on fire
            next_state - healthy/on_fire/burnt
            control - (delta_alpha, delta_beta)
        """
        if self.model == 'linear':
            return self.dynamics_linear(state_and_next_state, control)
        elif self.model == 'exponential':
            return self.dynamics_exponential(state_and_next_state, control)

    def dynamics_linear(self, state_and_next_state, control=(0, 0)):
        """
        Implementation of transition distribution. The transition from healthy to on fire
        is a linear function of the number of neighbors on fire.
        """

        state, number_neighbors_on_fire, next_state = state_and_next_state
        delta_alpha, delta_beta = control

        if state is self.healthy:
            if next_state is self.healthy:
                return 1 - (self.alpha - delta_alpha)*number_neighbors_on_fire
            elif next_state is self.on_fire:
                return (self.alpha - delta_alpha)*number_neighbors_on_fire
            else:
                return 0

        elif state is self.on_fire:
            if next_state is self.healthy:
                return 0
            elif next_state is self.on_fire:
                return self.beta - delta_beta
            elif next_state is self.burnt:
                return 1 - self.beta + delta_beta

        else:
            if next_state is self.burnt:
                return 1
            else:
                return 0

    def dynamics_exponential(self, state_and_next_state, control=(0, 0)):
        """
        Implementation of transition distribution. The transition from healty to on fire
        is an exponential function of the number of neighbors on fire.
        """

        state, number_neighbors_on_fire, next_state = state_and_next_state
        delta_alpha, delta_beta = control

        if state is self.healthy:
            if next_state is self.healthy:
                return (1 - self.alpha + delta_alpha)**number_neighbors_on_fire
            elif next_state is self.on_fire:
                return 1 - (1 - self.alpha + delta_alpha)**number_neighbors_on_fire
            else:
                return 0

        elif state is self.on_fire:
            if next_state is self.healthy:
                return 0
            elif next_state is self.on_fire:
                return self.beta - delta_beta
            elif next_state is self.burnt:
                return 1 - self.beta + delta_beta

        else:
            if next_state is self.burnt:
                return 1
            else:
                return 0

    # helper methods for querying/setting states
    def is_healthy(self, query):
        return query == self.healthy

    def is_on_fire(self, query):
        return query == self.on_fire

    def is_burnt(self, query):
        return query == self.burnt

    def set_on_fire(self):
        self.state = self.on_fire
