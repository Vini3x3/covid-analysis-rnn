import numpy as np


def lag_list(sequences: np.ndarray, period: int) -> np.ndarray:
    """
    Transform 2D sequence to a list of 2D delayed list
    :param sequences: input 2D sequence
    :param period: length of sub-sequence
    :return:
    """
    if sequences.shape[0] < period:
        raise LookupError("invalid period size")
    return np.array([sequences[i:i + period] for i in range(len(sequences) - period + 1)])


def moving_average(data: np.ndarray, window_size: int, alpha: float = 1.0):
    kernel = alpha ** np.arange(window_size - 1, -1, -1)
    kernel = kernel / sum(kernel)
    return np.convolve(data, kernel, mode='valid')

def normalize(vector: np.ndarray) -> np.ndarray:
    min_data = np.min(vector)
    max_data = np.max(vector)
    return (vector - min_data) / (max_data - min_data)

