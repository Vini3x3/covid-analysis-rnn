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
    if min_data == max_data:
        return np.zeros(vector.shape)
    return (vector - min_data) / (max_data - min_data)

def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    transposed = np.transpose(matrix)
    for i in range(transposed.shape[0]):
        transposed[i, :] = normalize(transposed[i, :])
    return np.transpose(transposed)


def diff_matrix(matrix: np.ndarray) -> np.ndarray:
    output_matrix = np.zeros(matrix.shape)
    for i in range(1, matrix.shape[-1]):
        output_matrix[i:, :] = matrix[i, :] - matrix[i - 1, :]
    return output_matrix

def diff(vector: np.ndarray) -> np.ndarray:
    output_vector = np.zeros_like(vector)
    output_vector[0] = 0
    output_vector[1:] = np.diff(vector)
    return output_vector

def transform_matrix(input_sequence: np.ndarray, mode: str = '', lag: int = 16) -> np.ndarray:
    if mode == 'MA':
        return moving_average(input_sequence, lag)
    elif mode == 'DMA':
        return moving_average(input_sequence, lag, 0.95)
    elif mode == 'D1':
        return np.diff(input_sequence)
    elif mode == 'NORM':
        return normalize_matrix(input_sequence)
    elif mode == 'DIFF':
        return diff_matrix(input_sequence)
    else:
        return input_sequence
