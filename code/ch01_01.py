'''
chapter 1.
타이타닉 대회 참여하기

- 데이터 불러오기
- 피처 추출
'''

import numpy as np
import pandas as pd

# 학습 데이터, 테스트 데이터의 불러오기
train = pd.read_csv('../data/ch01_titanic/train.csv')
test = pd.read_csv('../data/ch01_titanic/test.csv')

# 학습 데이터를 특징과 레이블로 나누기
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

# 테스트 데이터는 특징만 있으므로, 그대로 사용
test_x = test.copy()

# -----------------------------------
# 특징 추출(피처 엔지니어링)
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# 특징 PassengerId를 제거
train_x = train_x.drop(['PassengerId'], axis=1)
test_x = test_x.drop(['PassengerId'], axis=1)

# 특징 [Name, Ticket, Cabin]을 제거
train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 범주형 특징에 label encoding 을 적용하여 수치로 변환
for c in ['Sex', 'Embarked']:
    # 학습 데이터를 기반으로 어떻게 변환할지 최적화
    le = LabelEncoder()
    train_x[c] = train_x[c].fillna('NA')
    le.fit(train_x[c])

    # 학습 데이터, 테스트 데이터를 변환
    train_x[c] = le.transform(train_x[c])

    test_x[c] = test_x[c].fillna('NA')
    test_x[c] = le.transform(test_x[c].fillna('NA'))

# -----------------------------------
# 모델 만들기
# -----------------------------------
from xgboost import XGBClassifier

# 모델 생성 및 학습 데이터를 이용한 모델 학습
# model = XGBClassifier(n_estimators=20, random_state=71)
model = XGBClassifier(n_estimators=20, random_state=71, use_label_encoder=False)
model.fit(train_x, train_y)

# 테스트 데이터의 예측 결과를 확률로 출력
pred = model.predict_proba(test_x)[:, 1]

# 테스트 데이터의 예측 결과를 두개의 값(1,0)으로 변환
pred_label = np.where(pred > 0.5, 1, 0)

# 제출용 파일 작성
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)
# score ：0.7799（여기의 실행 결과가 사용자마다 다를 수 있을 가능성이 있습니다.）

# 데이터 전처리 파일 저장
train_x.to_csv('../data/ch01_titanic/train_x_01.csv', index=False)
test_x.to_csv('../data/ch01_titanic/test_x_01.csv', index=False)
train_y.to_csv('../data/ch01_titanic/train_y_01.csv', index=False)