import numpy as np

class Sigmoid:
    """
    Sigmoid function
    """
    def __call__(self, z):        
        return 1 / (1 + np.exp(-z))