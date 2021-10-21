# -----------------------------------
# 모델 튜닝
# -----------------------------------
#%%
import os
import itertools
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score

print(os.getcwd())
# 학습 데이터, 테스트 데이터의 불러오기
train_x = pd.read_csv('../data/ch01_titanic/train_x_01.csv')
train_y = pd.read_csv('../data/ch01_titanic/train_y_01.csv')
test_y = pd.read_csv('../data/ch01_titanic/test_x_01.csv')

# 튜닝을 위한 후보 파라미터 값을 준비
param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1.0, 2.0, 4.0]
}


#%%
# 하이퍼파라미터 값의 조합
param_combinations = itertools.product(param_space['max_depth'],
                                       param_space['min_child_weight'])

# print( list(param_combinations)[0:7] )

# 각 파라미터의 조합(params)과 그에 대한 점수를 보존(scores)하는 빈 리스트
params = []
scores = []

print( params, scores)
# 각 파라미터 조합별로 교차 검증(Cross-validation)으로 평가를 수행

#%%
for max_depth, min_child_weight in param_combinations:
    score_folds = []
    print("for start")
    
    # 교차 검증(Cross-validation)을 수행
    # 학습 데이터를 4개로 분할한 후,
    # 그중 하나를 평가용 데이터로 삼아 평가. 이를 데이터를 바꾸어 가면서 반복
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        # 학습 데이터를 학습 데이터와 평가용 데이터로 분할
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # 모델의 학습을 수행
        model = XGBClassifier(n_estimators=20, random_state=71,
                              use_label_encoder=False,
                              max_depth=max_depth,
                              min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)

        # 검증용 데이터의 점수를 계산한 후 저장
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        
        print(va_pred, logloss)
        score_folds.append(logloss)
        
    # 각 fold의 점수 평균을 구함
    score_mean = np.mean(score_folds)

    # 파라미터를 조합하고 그에 대한 점수를 저장
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

#%%
print(scores)
# 가장 점수가 좋은 것을 베스트 파라미터로 지정
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')
# max_depth=7, min_child_weight=2.0의 점수가 가장 좋음.