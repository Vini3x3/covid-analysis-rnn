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


