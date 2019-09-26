import yamlread as yr
import pandas as pd
from classifier import SA_Randomforest
from classifier import SA_Knn
from classifier import SA_xgboost
from classifier import SA_LinearRegression
import os
import datetime

def check_column_option(setting):
    if 'column_list' in setting["column_option_list"]:
        column_list = setting["column_option_list"]['column_list']
    else:
        column_list = None
    return column_list

def check_setting_file(setting):
    if not os.path.exists(setting["independent_path"]):
        raise FileNotFoundError(setting["independent_path"])
    if not os.path.exists(setting["dependent_file_path"]):
        raise FileNotFoundError(setting["dependent_file_path"])
    if not os.path.exists(setting["save_path"]):
        raise FileNotFoundError(setting["save_path"])


def check_classifier(setting, log_data):
    column_list = check_column_option(setting)

    if setting["classifier"] == "RF":
        # check option : HM4UP 과 같은 옵션 구별
        # coulums check : 컬럼설정한것 구별할 함수 있어야함
        # independent path 만 주고서 dataframe 파라미터를 빼도 될거 같음
        # type에 kospi 와 같은 것 넣는 이유 = save 시에 이름에 넣기 위해서
        #
        rf = SA_Randomforest(
            column_list=column_list,
            condition_list=setting["condition_list"],
            dependent_path=setting["dependent_file_path"],
            independent_path=setting["independent_path"],
            saved_path=setting["save_path"],
            start_date=setting["start_date"],
            seperate_date=setting["seperate_date"],
            end_date=setting["end_date"])

        if setting["column_option_list"]['option'] == "subset":
            rf.analyze(log=log_data,
                       n_estimators=setting["type_option_list"]["n_estimators"],
                       max_depth=int(setting["type_option_list"]["max_depth"]),
                       random_state=int(setting["type_option_list"]["random_state"])
                       )

        elif setting["column_option_list"]['option'] == "all":
            rf.analyze_auto(log=log_data,
                            range_of_column_no=setting["column_option_list"]["range_of_column_no"],
                            n_estimators=setting["type_option_list"]["n_estimators"],
                            max_depth=setting["type_option_list"]["max_depth"],
                            random_state=setting["type_option_list"]["random_state"])
        else:
            print("columns list option ERROR!!!")

    elif setting["classifier"] == "KNN":
        knn = SA_Knn(
            column_list=column_list,
            condition_list=setting["condition_list"],
            dependent_path=setting["dependent_file_path"],
            independent_path=setting["independent_path"],
            saved_path=setting["save_path"],
            start_date=setting["start_date"],
            seperate_date=setting["seperate_date"],
            end_date=setting["end_date"])

        if setting["column_option_list"]['option'] == "subset":
            knn.analyze(n_neighbors_list=setting["type_option_list"]["n_neighbors_list"],
                        log=log_data)

        elif setting["column_option_list"]['option'] == "all":
            knn.analyze_auto(log=log_data,
                             n_neighbors_list=setting["type_option_list"]["n_neighbors_list"],
                             range_of_column_no=setting["column_option_list"]["range_of_column_no"])
        else:
            print("columns list option ERROR!!!")

    elif setting["classifier"] == "LR":
        print("LR")
        lr = SA_LinearRegression(
            column_list=column_list,
            condition_list=setting["condition_list"],
            dependent_path=setting["dependent_file_path"],
            independent_path=setting["independent_path"],
            saved_path=setting["save_path"],
            start_date=setting["start_date"],
            seperate_date=setting["seperate_date"],
            end_date=setting["end_date"])

        if setting["column_option_list"]['option'] == "subset":
            lr.analyze(log=log_data)
        elif setting["column_option_list"]['option'] == "all":
            print("구현예정")

    elif setting["classifier"] == "xgboost":
        print("XGBoost")
        xgb = SA_xgboost(
            column_list=column_list,
            condition_list=setting["condition_list"],
            dependent_path=setting["dependent_file_path"],
            independent_path=setting["independent_path"],
            saved_path=setting["save_path"],
            start_date=setting["start_date"],
            seperate_date=setting["seperate_date"],
            end_date=setting["end_date"])

        if setting["column_option_list"]['option'] == "subset":
            xgb.analyze(log=log_data, n_estimators=100, min_child_weight=1, max_depth=8, gamma=0)

        elif setting["column_option_list"]['option'] == "all":
            xgb.analyze_auto(log=log_data,
                             range_of_column_no=setting["column_option_list"]["range_of_column_no"],
                             n_estimators=100, min_child_weight=1, max_depth=8, gamma=0)
        else:
            print("columns list option ERROR!!!")
    else:
        print("not right type!!")

if __name__ == '__main__':
    start_setting = yr.read_yaml_setting_value("finalsample.yaml")
    log_data = pd.DataFrame(columns=['classifier', 'condition', 'columns', 'accuracy',
                                     'precision', 'recall', 'margin', 'close_increase_rate', 'option'])
    for setting in start_setting['setting']:
        check_setting_file(setting)
        check_classifier(setting, log_data)
        print('-----------------------------------------------------------------')
    log_data.to_excel('log_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.xlsx')


# 추가해야할 것
# margin hm4up 말고 나머지들 추가
