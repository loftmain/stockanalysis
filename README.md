# stockanalysis

main.py를 실행하여 프로그램 실행합니다.  
현재 셋팅파일이 finalsample.yaml로 디폴트 되어있습니다.

yaml파일 형식은 밑에 설정과 같게 써야합니다.  
밑에 단락으로 들어갈 수록 띄어쓰기 4개씩 사용해야하며,  
띄어쓰기가 잘못된경우 오류발생을 하니 조심해야합니다.

 ### 기계학습 알고리즘 선택

    classifier: RF, KNN, LR, xgboost

 ### 알고리즘 셋팅 값 설정

-RF(randomforest)

     type_option_list:
        n_estimators: 100
        max_depth: 8
        random_state: 0

 -KNN

     type_option_list:
         n_neighbors_list: [3, 5, 7]

  -xgboost

    type_option_list:
        n_estimators=100
        min_child_weight=1
        max_depth=8
        gamma=0

 -LR(LinearRegression)-

     type_option_list:
         None

 ### 예측에 사용될 독립변수 설정

모든 독립변수 조합

    column_option_list:
        option: all
        range_of_column_no: [3, 4]

선택한 독립변수 조합

    column_option_list:
        option: subset
        column_list:
            - ["PB", "SATP"]
            - ['MORTGAGE30US', 'SATP', 'ERA', 'USSLIND']

 ### 예측 되어질 종속변수 설정(여러개 가능)

    condition_list:
        - HM4UP
        - HM3UP

 ### 독립변수 파일 경로

    dependent_file_path: dependent/kospi_xgb_test_dependent.xlsx

 ### 종속변수 폴더 경로

    independent_path: xgb_sample_independent

 ### 결과 저장 경로

    save_path: save

 ### 분석 시작, 분기점, 종료 지정

    start_date: 2009-09-01
    seperate_date: 2016-09-01
    end_date: 2019-04-01
