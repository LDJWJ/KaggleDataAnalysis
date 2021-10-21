'''
chapter 1. 타이타닉 대회 참여하기
- 모델 검증
 * logloss, accuracy 구하기
'''

# -----------------------------------
# 모델 검증
# -----------------------------------
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

# 학습 데이터, 테스트 데이터의 불러오기
train_x = pd.read_csv('../data/ch01_titanic/train_x_01.csv')
train_y = pd.read_csv('../data/ch01_titanic/train_y_01.csv')
test_y = pd.read_csv('../data/ch01_titanic/test_x_01.csv')

# 각 fold의 평가 점수를 저장을 위한 빈 리스트 선언
scores_accuracy = []
scores_logloss = []

# 교차 검증(Cross-validation)을 수행
# 01 학습 데이터를 5개로 분할
# 02 그중 하나를 평가용 데이터셋으로 지정
# 03 이후 평가용 데이터의 블록을 하나씩 옆으로 옮겨가며 검증을 수행
kf = KFold(n_splits=5, shuffle=True, random_state=71)  # 4 -> 5
for tr_idx, va_idx in kf.split(train_x):
    # 학습 데이터를 학습 데이터와 평가용 데이터셋으로 분할
    # tr_idx, va_idx는 각각의 데이터의 인덱스 값.
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
# logloss: 0.4270, accuracy: 0.8148 (결과가 다를 수 있음.)
