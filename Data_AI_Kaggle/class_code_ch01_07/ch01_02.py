# -*- coding: utf-8 -*-
"""
타이타닉 데이터 셋을 활용한 데이터 처리 및 모델 생성
"""
import os
import numpy as np
import pandas as pd

print(os.getcwd())
os.chdir('C:/Users/toto/Documents/Github/KaggleDataAnalysis/Data_AI_Kaggle/class_code_ch01_07')
print(os.getcwd())


# 학습 데이터, 테스트 데이터의 불러오기
train_x = pd.read_csv('train_x.csv')
test_y = pd.read_csv('test_x.csv')


# -----------------------------------
# 모델 검증
# -----------------------------------
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

# 각 fold의 평가 점수를 저장을 위한 빈 리스트 선언
scores_accuracy = []
scores_logloss = []

# 교차 검증(Cross-validation)을 수행
# 01 학습 데이터를 4개로 분할
# 02 그중 하나를 평가용 데이터셋으로 지정
# 03 이후 평가용 데이터의 블록을 하나씩 옆으로 옮겨가며 검증을 수행
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    # 학습 데이터를 학습 데이터와 평가용 데이터셋으로 분할
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 모델 학습을 수행
    # model = XGBClassifier(n_estimators=20, random_state=71)
    model = XGBClassifier(n_estimators=20, 
                          random_state=71, 
                          use_label_encoder=False)
    
    model.fit(tr_x, tr_y)

    # 평가용 데이터의 예측 결과를 확률로 출력
    va_pred = model.predict_proba(va_x)[:, 1]

    # 평가용 데이터의 점수를 계산
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    # 각 fold의 점수를 저장
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)
    

#각 fold의 점수 평균을 출력.
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')

# -----------------------------------
# 모델 튜닝
# -----------------------------------
import itertools

# 튜닝을 위한 후보 파라미터 값을 준비
param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1.0, 2.0, 4.0]
}

# 하이퍼파라미터 값의 조합
param_combinations = itertools.product(param_space['max_depth'], 
                                       param_space['min_child_weight'])

print(list(param_combinations) )

# 각 파라미터의 조합(params)과 그에 대한 점수를 보존(scores)하는 빈 리스트
params = []
scores = []

# 각 파라미터 조합별로 교차 검증(Cross-validation)으로 평가를 수행
for max_depth, min_child_weight in param_combinations:

    score_folds = []
    
    # 교차 검증(Cross-validation)을 수행
    # 학습 데이터를 4개로 분할한 후,
    # 그중 하나를 평가용 데이터로 삼아 평가. 이를 데이터를 바꾸어 가면서 반복
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    
    for tr_idx, va_idx in kf.split(train_x):
        # 학습 데이터를 학습 데이터와 평가용 데이터로 분할
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # 모델의 학습을 수행
        model = XGBClassifier(n_estimators=20, 
                              random_state=71, 
                              use_label_encoder=False,
                              max_depth=max_depth, 
                              min_child_weight=min_child_weight)
        
        model.fit(tr_x, tr_y)

        # 검증용 데이터의 점수를 계산한 후 저장
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        score_folds.append(logloss)

    # 각 fold의 점수 평균을 구함
    score_mean = np.mean(score_folds)

    # 파라미터를 조합하고 그에 대한 점수를 저장
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

# 가장 점수가 좋은 것을 베스트 파라미터로 지정
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]

print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')
# max_depth=7, min_child_weight=2.0의 점수가 가장 좋음.

