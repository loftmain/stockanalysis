import yamlread as yr
from classifier import SA_Randomforest
from classifier import SA_Knn

def check_column_option(start_setting, index_):
    if 'column_list' in start_setting["column_option_list"][index_]:
        column_list = start_setting["column_option_list"][index_]['column_list']
    else:
        column_list = None
    return column_list

def check_classifier(type_, start_setting, index_):
    column_list = check_column_option(start_setting, index_)

    if type_ == "RF":
        # check option : HM4UP 과 같은 옵션 구별
        # coulums check : 컬럼설정한것 구별할 함수 있어야함
        # independent path 만 주고서 dataframe 파라미터를 빼도 될거 같음
        # type에 kospi 와 같은 것 넣는 이유 = save 시에 이름에 넣기 위해서
        #
        rf = SA_Randomforest(
            type=start_setting["type_list"][index_],
            column_list=column_list,
            condition_list=start_setting["condition_list"][index_],
            dependent_path=start_setting["dependent_path_list"][index_],
            independent_path=start_setting["independent_path"][index_],
            saved_path=start_setting["save_path"][index_],
            start_date=start_setting["start_date"],
            seperate_date=start_setting["seperate_date"],
            end_date=start_setting["end_date"])

        if start_setting["column_option_list"][index_]['option'] == "partition":
            rf.analyze(n_estimators=100, max_depth=None, random_state=0)

        elif start_setting["column_option_list"][index_]['option'] == "all":
            rf.analyze_auto(n_estimators=100, max_depth=None, random_state=0)
        else:
            print("columns list option ERROR!!!")

    elif type_ == "KNN":
        knn = SA_Knn(type=start_setting["type_list"][index_],
                     column_list=column_list,
                     condition_list=start_setting["condition_list"][index_],
                     dependent_path=start_setting["dependent_path_list"][index_],
                     independent_path=start_setting["independent_path"][index_],
                     saved_path=start_setting["save_path"][index_],
                     start_date=start_setting["start_date"],
                     seperate_date=start_setting["seperate_date"],
                     end_date=start_setting["end_date"])

        if start_setting["column_option_list"][index_]['option'] == "partition":
            knn.analyze(start_setting["type_option_list"][index_]["n_neighbors_list"])
        elif start_setting["column_option_list"][index_]['option'] == "all":
            #knn.analyze_auto(start_setting["type_option_list"][index_]["n_neighbors_list"])
        else:
            print("columns list option ERROR!!!")
    elif type_ == "LR":
        print("LR")

    elif type_ == "XGBoost":
        print("XGBoost")

    else:
        print("not right type!!")

if __name__ == '__main__':
    start_setting = yr.read_yaml_setting_value("yaml.yaml")
    for i, type_ in enumerate(start_setting['classifier_list']):
        check_classifier(type_, start_setting, i)

        print('-----------------------------------------------------------------')
    print(start_setting["column_option_list"][2]["column_max_count"])

