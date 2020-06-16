import numpy as np

class Activation:
    @staticmethod
    def of(x: float):
        raise NotImplementedError

    @staticmethod
    def prime(x: float):
        raise NotImplementedError

class Sigmoid(Activation):

    @staticmethod
    def of(x: float):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def prime(x: float):
        return Sigmoid.of(x) * (1 - Sigmoid.of(x))

class Relu(Activation):
    @staticmethod
    def of(x: float):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def prime(x: float):
        return Sigmoid.of(x) * (1.0 - Sigmoid.of(x))