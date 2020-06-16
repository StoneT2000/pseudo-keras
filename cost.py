import numpy as np

from activation import Sigmoid

class Cost:
    def __init__(self):
        pass
    @staticmethod
    def fn(a, y):
        """
        `a` is the activation
        `y` is the desired output
        """
        raise NotImplementedError
    @staticmethod
    def delta(z, a, y):
        """
        `z` is the weighted input
        `a` is the activation
        `y` is the desired output
        """
        raise NotImplementedError

class QuadraticCost(Cost):
    def __init__(self):
        pass
    @staticmethod
    def fn(a, y):
        0.5 * np.linalg.norm(a - y) ** 2
    @staticmethod
    def delta(z, a, y):
        return (a-y) * Sigmoid.prime(z)

class CrossEntropyCost(Cost):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
    @staticmethod
    def delta(z, a, y):
        return a - y