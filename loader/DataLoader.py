import numpy as np
import pandas as pd

from lib.covid_module import get_date_count


def read_dataframe(name: str) -> pd.DataFrame:
    if name == 'infected':
        df_infected = pd.read_csv("data/covid/covid_hk_case_std.csv")
        df_infected['report_date'] = pd.to_datetime(df_infected['report_date'], format='%Y%m%d')
        return df_infected
    else:
        raise NotImplementedError()


def read_sequence(name: str) -> np.ndarray:
    if name == 'infected':
        df = read_dataframe(name)
        return get_date_count(df, 'report_date', '%Y%m%d')['count'].to_numpy()
    else:
        raise NotImplementedError()
