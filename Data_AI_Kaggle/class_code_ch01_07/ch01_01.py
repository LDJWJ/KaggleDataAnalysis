# -*- coding: utf-8 -*-
"""
타이타닉 데이터 셋을 활용한 데이터 처리 및 모델 생성
"""
import os
import numpy as np
import pandas as pd

os.chdir('C:/Users/toto/Documents/Github/KaggleDataAnalysis/Data_AI_Kaggle/class_code_ch01_07')
print(os.getcwd())

# 학습 데이터, 테스트 데이터의 불러오기
train = pd.read_csv('../input/ch01-titanic/train.csv')
test = pd.read_csv('../input/ch01-titanic/test.csv')

# 데이터를 입력과 레이블로 분리
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

test_x = test.copy()

# 데이터 전처리
print(train_x.info())
train_x['Embarked'] = train_x['Embarked'].fillna("S")

# 03 특징 추출
# 특징 제거
sel = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_x = train_x.drop(sel, axis=1)
test_x = test_x.drop(sel, axis=1)

mapping1 = {"male":1 , "female":2}
train_x['Sex'] = train_x['Sex'].map(mapping1)
test_x['Sex'] = test_x['Sex'].map(mapping1)

mapping2 = {"C":1 , "Q":2, "S":3}
train_x['Embarked'] = train_x['Embarked'].map(mapping2)
test_x['Embarked'] = test_x['Embarked'].map(mapping2)

### 04 모델 생성
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=20, 
                      random_state=71, 
                      use_label_encoder=False)

model.fit(train_x, train_y)

# 테스트 데이터의 예측 결과를 확률로 출력
pred = model.predict_proba(test_x)[:, 1]

# 테스트 데이터의 예측 결과를 두개의 값(1,0)으로 변환
pred_label = np.where(pred > 0.5, 1, 0)

# 제출용 파일 작성
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 
                           'Survived': pred_label})
submission.to_csv('submission_first.csv', index=False)

# 전처리 후, 데이터 생성
train_x.to_csv("train_x.csv", index=False)
test_x.to_csv("test_x.csv", index=False)




