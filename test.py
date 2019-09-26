import yamlread as yr
import pandas as pd
from io import StringIO
import os
import numpy as np

from datetime import datetime
class SuperError(Exception):

    def __init__(self, msg):
        super().__init__(msg)



if __name__ == '__main__':

    start_setting = yr.read_yaml_setting_value("xgb_sample.yaml")
    """
    #for i, type_ in enumerate(start_setting['start']):
    #    print(type_)
    print(start_setting['setting'][0]["classifier"])
    print(start_setting['setting'][0]["type_option_list"])
    print(start_setting['setting'][0]["column_option_list"]["option"])
    print(start_setting['setting'][0]["column_option_list"]["column_list"])
    print(start_setting['setting'][0]["condition_list"])
    print(start_setting['setting'][0]["dependent_path_list"])
    print(start_setting['setting'][0]["independent_path"])
    print(start_setting['setting'][0]["save_path"])

    print("-----------------------------------------------------")

    print(start_setting['setting'][1]["classifier"])
    print(start_setting['setting'][1]["type_option_list"])
    print(start_setting['setting'][1]["column_option_list"]["option"])
    print(start_setting['setting'][1]["column_option_list"])
    print(start_setting['setting'][1]["condition_list"])
    print(start_setting['setting'][1]["dependent_path_list"])
    print(start_setting['setting'][1]["independent_path"])
    print(start_setting['setting'][1]["save_path"])
    #if start_setting['setting'][1]["column_option_list"]['option'][0] == "all":
    #    for i in start_setting['setting'][1]["column_option_list"]['option']:
    #        print(i)
    for i, j in start_setting['setting'][1]["type_option_list"].items():
        print(i, j)
    """
    csv_data = """A,B,C,D,
    1.,2.,3.,4.,
    5.,6.,,8."""
    df=pd.read_csv(StringIO(csv_data), encoding='UTF-8')
    df = df.drop(['Unnamed: 4'], axis=1)
    df = df.replace('', np.nan)
    print(df)
    df_na = df.isnull()
    df_na_sum = df.isnull().sum().sum()
    print(df_na_sum)
    dict()