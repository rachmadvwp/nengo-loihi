import numpy as np


class Profile(object):
    def __eq__(self, obj):
        return isinstance(obj, type(self)) and all(
            self.__dict__[key] == obj.__dict__[key] for key in self.params)

    def __hash__(self):
        return hash(tuple(self.__dict__[key] for key in self.params))


def shift(x, s, **kwargs):
    if s < 0:
        return np.right_shift(x, -s, **kwargs)
    else:
        return np.left_shift(x, s, **kwargs)
