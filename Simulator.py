import warnings


class Simulator(object):
    def __init__(self):
        self.dims = None
        self.group = None
        self.iter = 0
        self.end = False

    def reset(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def dense_state(self):
        raise NotImplementedError
