import os

import numpy as np
import pandas as pd

from lib.covid_module import get_date_count
from loader import DataTransformer


def read_dataframe(name: str) -> pd.DataFrame:
    curr_dir = os.getcwd()
    project_dir = curr_dir.split('GitHub')[0]
    analysis_on_covid_dir = os.path.join(project_dir, 'GitHub', 'analysis-on-covid')
    if name == 'case':
        df_case = pd.read_csv(analysis_on_covid_dir + "/data/std_data/hk/covid_hk_case_detail_std.csv")
        df_case['report_date'] = pd.to_datetime(df_case['report_date'], format='%Y%m%d')
        return df_case
    if name == 'count':
        df_case = pd.read_csv(analysis_on_covid_dir + "/data/std_data/hk/covid_hk_case_count_std.csv")
        df_case['report_date'] = pd.to_datetime(df_case['report_date'], format='%Y%m%d')
        data = {
            'report_date': df_case['report_date'],
            'count': DataTransformer.diff(df_case['cuml_case_cnt'].to_numpy())
        }
        return pd.DataFrame(data)
    elif name == 'temp':
        df_temp = pd.read_csv(analysis_on_covid_dir + "/data/std_data/hk/hk_daily_avg_temp_std.csv")
        df_temp['report_date'] = pd.to_datetime(df_temp['report_date'], format='%Y%m%d')
        return df_temp
    elif name == 'vacc':
        df_vacc = pd.read_csv(analysis_on_covid_dir + "/data/std_data/hk/covid_hk_vacc_daily_count_std.csv")
        df_vacc['report_date'] = pd.to_datetime(df_vacc['report_date'], format='%Y%m%d')
        return df_vacc
    elif name == 'policy':
        df_policy = pd.read_csv(analysis_on_covid_dir + '/data/std_data/hk/covid_hk_policy_std.csv')
        df_policy['report_date'] = pd.to_datetime(df_policy['report_date'], format='%Y%m%d')
        return df_policy
    elif name == 'humid':
        df_humid = pd.read_csv(analysis_on_covid_dir + '/data/std_data/hk/hk_daily_avg_humid_std.csv')
        df_humid['report_date'] = pd.to_datetime(df_humid['report_date'], format='%Y%m%d')
        return df_humid
    elif name == 'vac_age':
        df_vac_age = pd.read_csv(analysis_on_covid_dir + '/data/std_data/hk/hk_vacc_age_grp_daily_count.csv')
        df_vac_age['report_date'] = pd.to_datetime(df_vac_age['report_date'], format='%Y%m%d')
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
    # count
    df_count = read_dataframe('count')

    # temp
    df_temp = read_dataframe('temp')
    df_temp = df_temp[['report_date', 'avg_temp']]

    # vacc
    df_vacc = read_dataframe('vacc')
    df_vacc = df_vacc[['report_date',
                       'sinov_1st_dose', 'sinov_2nd_dose', 'sinov_3rd_dose',
                       'biont_1st_dose', 'biont_2nd_dose', 'biont_3rd_dose']]
    df_vacc = get_date_sum(df_vacc, 'report_date', '%Y%m%d')

    df_policy = read_dataframe('policy')

    df_humid = read_dataframe('df_humid')
    df_humid = df_humid[['report_date', 'avg_humid']]

    # merge
    df_all = pd.merge_asof(df_count, df_temp, on="report_date", direction='backward')  # left join
    df_all = pd.merge_asof(df_all, df_vacc, on='report_date', direction='backward')  # left join
    df_all = pd.merge_asof(df_all, df_policy, on='report_date', direction='backward')  # left join
    df_all = pd.merge_asof(df_all, df_humid, on='report_date', direction='backward')  # left join
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
