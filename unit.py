import numpy as np

from loader.DataLoader import read_dataframe
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


def t3_read_dataframe():
    df_case = read_dataframe('case')
    assert list(df_case.columns) == [
        'report_date', 'onset_date', 'gender', 'age', 'case_outcome', 'resident', 'case_type', 'case_status',
        'age_group', 'import_local', 'report_year', 'report_year_month'
    ]

    df_temp = read_dataframe('temp')
    assert list(df_temp.columns) == [
        'report_date', 'report_year', 'report_month', 'report_day', 'avg_temp', 'min_temp', 'max_temp'
    ]

    df_vaac = read_dataframe('vacc')
    assert list(df_vaac.columns) == [
        'report_date', 'age_group', 'gender', 'sinov_1st_dose', 'sinov_2nd_dose', 'sinov_3rd_dose', 'sinov_4th_dose',
        'sinov_5th_dose', 'sinov_6th_dose', 'sinov_7th_dose', 'biont_1st_dose', 'biont_2nd_dose', 'biont_3rd_dose',
        'biont_4th_dose', 'biont_5th_dose', 'biont_6th_dose', 'biont_7th_dose', 'does_all', 'sinov_1st_dose_cum',
        'sinov_2nd_dose_cum', 'sinov_3rd_dose_cum', 'sinov_4th_dose_cum', 'sinov_5th_dose_cum', 'sinov_6th_dose_cum',
        'sinov_7th_dose_cum', 'biont_1st_dose_cum', 'biont_2nd_dose_cum', 'biont_3rd_dose_cum', 'biont_4th_dose_cum',
        'biont_5th_dose_cum', 'biont_6th_dose_cum', 'biont_7th_dose_cum', 'does_all_cum', 'report_year',
        'report_year_month'
    ]


def t4_read_all_df():
    df_all = read_dataframe('all')
    assert (set(df_all.columns)) == {'report_date', 'avg_temp', 'min_temp', 'max_temp', 'count', 'sum'}


if __name__ == '__main__':
    t1_delay_happy()
    t2_ma()
    t3_read_dataframe()
    t4_read_all_df()
