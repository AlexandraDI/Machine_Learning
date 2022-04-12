import numpy as np


class Dataset:
    """
    Generate the data for the assignment

    Args:
        size: number of features of each sample
    """

    def __init__(self, size: int = 8):
        self._data = np.eye(8)

    def get_data(self, shuffle: bool = False) -> np.ndarray:
        """
        Get the input samples for the assignment

        Args:
            shuffle: whether to shuffle the list of samples
        Returns:
            Matrix containin the samples
        """
        tmp = self._data.copy()
        if shuffle:
            return tmp[:, np.random.permutation(tmp.shape[1])]  # re-arrange the columns
        else:
            return tmp
        
    def get_other_data(self, size: int = 8, p_one: float = 0.5) -> np.ndarray:
        """
        Get vectors with more than one 1.
        
        Args:
            size: the number of samples to generate
            p_one: the probability of having 1 on the vector (e.g., with 0.5, we expect 4 1s and 4 0s)
        Returns:
            Matrix containing the samples
        """
        return np.random.rand(8, size) <= p_one

    def get_other_data_test(self, size: int = 8, p_one: float = 0.1) -> np.ndarray:
        """
        Get vectors with more than one 1.
        
        Args:
            size: the number of samples to generate
            p_one: the probability of having 1 on the vector (e.g., with 0.5, we expect 4 1s and 4 0s)
        Returns:
            Matrix containing the samples
        """
        return np.random.rand(8, size) <= p_one