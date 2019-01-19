import warnings


class Element(object):
    def __init__(self):
        self.state = None
        self.next_state = None

    def reset(self):
        raise NotImplementedError

    def next(self, forest, control):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
