import numpy as np

from loader.DataTransformer import lag_list, moving_average


def t1_delay_happy():
    input_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    output_arr = lag_list(input_arr, 2)
    assert output_arr.shape == (4, 2, 3)
    assert (output_arr[0][0] == [1, 2, 3]).all()
    assert (output_arr[-1][-1] == [13, 14, 15]).all()

def t2_ma():
    input_arr = np.array(list(range(10)))
    window = 4

    output_arr_1 = moving_average(input_arr, window, 1)

    kernel = np.ones(window) / window
    output_arr_2 = np.convolve(input_arr, kernel, mode='valid')

    assert (output_arr_1 == output_arr_2).all()

if __name__ == '__main__':
    t1_delay_happy()
    t2_ma()