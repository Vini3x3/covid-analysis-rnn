import numpy as np

from loader.DataTransformer import lag_list


def t1_delay_happy():
    input_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    output_arr = lag_list(input_arr, 2)
    assert output_arr.shape == (4, 2, 3)
    assert (output_arr[0][0] == [1, 2, 3]).all()
    assert (output_arr[-1][-1] == [13, 14, 15]).all()

if __name__ == '__main__':
    t1_delay_happy()