from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import chain
from operator import eq
import numpy as np


def identity(obj):
    """Returns directly the argument *obj*.
    """
    return obj

class Statistics(object):
    """Object that compiles statistics on a list of arbitrary objects.
    When created the statistics object receives a *key* argument that
    is used to get the values on which the function will be computed.
    If not provided the *key* argument defaults to the identity function.

    The value returned by the key may be a multi-dimensional object, i.e.:
    a tuple or a list, as long as the statistical function registered
    support it. So for example, statistics can be computed directly on
    multi-objective fitnesses when using numpy statistical function.

    :param key: A function to access the values on which to compute the
                statistics, optional.
    """
    def __init__(self, key=identity):
        self.key = key
        self.functions = dict()
        self.fields = []

    def register(self, name, function, *args, **kargs):
        """Register a *function* that will be applied on the sequence each
        time :meth:`record` is called.

        :param name: The name of the statistics function as it would appear
                     in the dictionary of the statistics object.
        :param function: A function that will compute the desired statistics
                         on the data as preprocessed by the key.
        :param argument: One or more argument (and keyword argument) to pass
                         automatically to the registered function when called,
                         optional.
        """
        self.functions[name] = partial(function, *args, **kargs)
        self.fields.append(name)

    def compile(self, data):
        """Apply to the input sequence *data* each registered function
        and return the results as a dictionary.

        :param data: Sequence of objects on which the statistics are computed.
        """
        values = tuple(self.key(elem) for elem in data)
        entry = []
        for _, func in iter(self.functions.items()):
            entry.append(func(values))
        return entry




