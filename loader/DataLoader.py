import numpy as np
import pandas as pd

from lib.covid_module import get_date_count


def read_dataframe(name: str) -> pd.DataFrame:
    if name == 'case':
        df_case = pd.read_csv("data/covid/covid_hk_case_std.csv")
        df_case['report_date'] = pd.to_datetime(df_case['report_date'], format='%Y%m%d')
        return df_case
    elif name == 'temp':
        df_temp = pd.read_csv("data/covid/hk_daily_temp_std.csv")
        df_temp['report_date'] = pd.to_datetime(df_temp['report_date'], format='%Y%m%d')
        return df_temp
    elif name == 'vacc':
        df_vacc = pd.read_csv("data/covid/covid_hk_vacc_std.csv")
        df_vacc['report_date'] = pd.to_datetime(df_vacc['report_date'], format='%Y%m%d')
        return df_vacc
    elif name == 'all':
        return read_join_df()
    else:
        raise NotImplementedError()


def read_sequence(name: str) -> np.ndarray:
    if name == 'case':
        df = read_dataframe(name)
        return get_date_count(df, 'report_date', '%Y%m%d')['count'].to_numpy()
    else:
        raise NotImplementedError()


def read_join_df() -> pd.DataFrame:
    # case
    df_case = read_dataframe('case')
    df_case = get_date_count(df_case, 'report_date', '%Y%m%d')

    # temp
    df_temp = read_dataframe('temp')
    df_temp = df_temp[['report_date', 'avg_temp', 'min_temp', 'max_temp']]

    # vacc
    df_vacc = read_dataframe('vacc')
    df_vacc = df_vacc[[
        'report_date', 'sinov_1st_dose', 'sinov_2nd_dose', 'sinov_3rd_dose', 'sinov_4th_dose',
        'sinov_5th_dose', 'sinov_6th_dose', 'sinov_7th_dose', 'biont_1st_dose', 'biont_2nd_dose', 'biont_3rd_dose',
        'biont_4th_dose', 'biont_5th_dose', 'biont_6th_dose', 'biont_7th_dose', 'does_all', 'sinov_1st_dose_cum',
        'sinov_2nd_dose_cum', 'sinov_3rd_dose_cum', 'sinov_4th_dose_cum', 'sinov_5th_dose_cum', 'sinov_6th_dose_cum',
        'sinov_7th_dose_cum', 'biont_1st_dose_cum', 'biont_2nd_dose_cum', 'biont_3rd_dose_cum', 'biont_4th_dose_cum',
        'biont_5th_dose_cum', 'biont_6th_dose_cum', 'biont_7th_dose_cum'
    ]]
    df_vacc = get_date_sum(df_vacc, 'report_date', '%Y%m%d')

    # merge
    df_all = pd.merge_asof(df_case, df_temp, on="report_date", direction='backward')  # left join
    df_all = pd.merge_asof(df_all, df_vacc, on='report_date', direction='backward')  # left join
    df_all = df_all.fillna(0)

    # rearrange columns
    rearranged_columns = list(df_all.columns)
    rearranged_columns.remove('count')
    rearranged_columns.append('count')
    df_all = df_all[rearranged_columns]

    return df_all


def get_date_sum(df: pd.DataFrame, col: str, date_format: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
    agg_df = df.groupby(col).sum()
    agg_df['sum'] = agg_df.sum(axis=1, numeric_only=True)
    return agg_df[['sum']].reset_index()
