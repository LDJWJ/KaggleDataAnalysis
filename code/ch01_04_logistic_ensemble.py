# -----------------------------------
# 로지스틱 회귀용 특징 작성
# -----------------------------------
#%%
from sklearn.preprocessing import OneHotEncoder

# 학습 데이터, 테스트 데이터의 불러오기
train = pd.read_csv('../data/ch01_titanic/train.csv')
test = pd.read_csv('../data/ch01_titanic/test.csv')

# 원본 데이터를 복사하기
train_x2 = train.drop(['Survived'], axis=1)
test_x2 = test.copy()

# 특징 PassengerId를 제거
train_x2 = train_x2.drop(['PassengerId'], axis=1)
test_x2 = test_x2.drop(['PassengerId'], axis=1)

# 특징 [Name, Ticket, Cabin]을 제거
train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 원핫 인코딩(one hot encoding)을 수행
cat_cols = ['Sex', 'Embarked', 'Pclass']
ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(train_x2[cat_cols].fillna('NA'))

print(type(ohe.categories_) , ohe.categories_)

#%%
# 원핫 인코딩의 더미 변수의 열이름을 작성
ohe_columns = []
for i, c in enumerate(cat_cols):
    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]

print(ohe_columns)

#%%
# 원핫 인코딩에 의한 변환을 수행
ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)
ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)

# 원핫 인코딩이 수행 후, 원래 특징를 제거
train_x2 = train_x2.drop(cat_cols, axis=1)
test_x2 = test_x2.drop(cat_cols, axis=1)

# 원핫 인코딩을 수행된 특징를 결합
train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)
test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)

# 수치변수의 결측치를 학습 데이터의 평균으로 채우기
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
for col in num_cols:
    train_x2[col].fillna(train_x2[col].mean(), inplace=True)
    test_x2[col].fillna(train_x2[col].mean(), inplace=True)

# 특징Fare를 로그 변환을 수행
train_x2['Fare'] = np.log1p(train_x2['Fare'])
test_x2['Fare'] = np.log1p(test_x2['Fare'])


#%%
# -----------------------------------
# 앙상블(ensemble)
# -----------------------------------
from sklearn.linear_model import LogisticRegression

# xgboost 모델
model_xgb = XGBClassifier(n_estimators=20, random_state=71, use_label_encoder=False)
model_xgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]

# 로지스틱 회귀 모델
# xgboost 모델과는 다른 특징을 넣어야 하므로 train_x2, test_x2를 생성
model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
model_lr.fit(train_x2, train_y)
pred_lr = model_lr.predict_proba(test_x2)[:, 1]

# 예측 결과의 가중 평균 구하기
pred = pred_xgb * 0.8 + pred_lr * 0.2
pred_label = np.where(pred > 0.5, 1, 0)

# 제출용 파일 작성
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})
submission.to_csv('submission_first_ensemble.csv', index=False)
# score ：0.7799（여기의 실행 결과가 사용자마다 다를 수 있을 가능성이 있습니다.）