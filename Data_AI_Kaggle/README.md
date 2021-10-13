# kagglebook

책 제목 : 데이터가 뛰어노는 AI 놀이터, 캐글
책 링크 : 
 * https://www.hanbit.co.kr/store/books/look.php?p_code=B4998513859 (한빛 미디어)
 * http://www.yes24.com/Product/Goods/101479127?OzSrank=1 (yes24)
 * https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=270739663 (알라딘)

(독자 의견) frontier1020@naver.com
 * 책과 기타 소중한 의견이 있으시면 메일을 주세요~

### 이 책에서의 라이브러리 버전 정보(실행 확인 2021.10)
 - 파이썬 버전 :  3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]
 - 팬더스 버전 :  1.2.4
 - matplotlib 버전 :  3.3.4
 - 넘파이 버전 :  1.19.5
 - scikit-learn 버전 :  0.24.1
 - tensorflow 버전 :  2.6.0
 - 케라스 버전 :  2.6.0
 - xgboost 버전 :  1.4.2
 - lightgbm 버전 :  3.3.0
 - hyperopt 버전 :  0.2.5
 - umap-learn 버전 :  0.5.1  # umap (2021.10)

 - 일부 라이브러리 버전이 맞지 않을 경우, 소스코드 일부가 실행이 되지 않을 수 있습니다. 
 - 최근에 tensorflow의 버전이 2.6.x로 변경되었습니다. 일부 실행이 되지 않는다면 버전에 맞춰 재 설치를 부탁드립니다(2021/10/13)

### 에러 등의 업데이트 [상세 내용 확인하기](./pdf_html/issue_list.html)
 - 01 keras 설치 후, 불러올때 에러 발생(2021/06/02 추가)

### 환경 준비(Window용) 
 - [설치하기 설명(simple버전)-가상 환경 설치없음](https://ldjwj.github.io/kagglebook/pdf_html/1_1_env_simple.html)
 - [설치하기 설명-가상 환경 설치](https://ldjwj.github.io/kagglebook/pdf_html/1_1_env.html)

### 환경준비(MAC용)
 - (추가 예정)
### 환경준비(Linux용)
 - (추가 예정)

### 일부 유저에 소스 코드 실행 이슈가 있음.
 - bhtsne 모듈 설치가 안되는 이슈(2021.03.13) - 버전 불일치로 판단됨.

## 소스 코드 보기
 ### 수업용 코드로 이동 [Link](https://github.com/LDJWJ/KaggleDataAnalysis/tree/main/Data_AI_Kaggle/class_code_ch01_07)
 
 ### ch01 - 타이타닉 대회

 ### ch02
   * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb)|
|------|---|---|
|ch02-01-metrics        |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch02/ch02-01-metrics.py)|[CODE(노트북)] |
|ch02-02-custom-usage   |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch02/ch02-02-custom-usage.py)|[CODE(노트북)]|
|ch02-03-optimize       |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch02/ch02-03-optimize.py) |[CODE(노트북)]|
|ch02-04-optimize-cv    |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch02/ch02-04-optimize-cv.py)|[CODE(노트북)]|
|ch02-05-custom-function|[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch02/ch02-05-custom-function.py)| [CODE(노트북)]|

 ### ch03
   * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb)|
|------|---|---|
|ch03-01-numerical.py      |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch03/ch03-01-numerical.py)|[CODE(노트북)]|
|ch03-02-categorical.py    |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch03/ch03-02-categorical.py)|[CODE(노트북)]|
|ch03-03-multi_tables.py   |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch03/ch03-03-multi_tables.py) |[CODE(노트북)]|
|ch03-04-time_series.py    |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch03/ch03-04-time_series.py)|[CODE(노트북)]|
|ch03-05-reduction.py      |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch03/ch03-05-reduction.py)| [CODE(노트북)]|
|ch03-06-reduction-mnist.py|[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch03/ch03-06-reduction-mnist.py)| [CODE(노트북)]|

 ### ch04
   * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb)|
|------|---|---|
|ch04-01-introduction.py      |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch04/ch03-01-numerical.py)|[CODE(노트북)]|
|ch04-02-run_xgb.py   |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch04/ch04-02-run_xgb.py)|[CODE(노트북)]|
|ch04-03-run_lgb.py   |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch04/ch04-03-run_lgb.py) |[CODE(노트북)]|
|ch04-04-run_nn.py    |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch04/ch04-04-run_nn.py)|[CODE(노트북)]|
|ch04-05-run_linear.py      |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch04/ch04-05-run_linear.py)| [CODE(노트북)]|


 ### ch05
    * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb)|
|------|---|---|
|ch05-01-validation.py      |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch05/ch05-01-validation.py)|[CODE(노트북)]|
|ch05-02-timeseries.py   |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch05/ch05-02-timeseries.py)|[CODE(노트북)]|

 ### ch06
   * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb)|
|------|---|---|
|ch06-01-hopt.py         |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-01-hopt.py)|[CODE(노트북)]|
|ch06-02-hopt_xgb.py     |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-02-hopt_xgb.py)|[CODE(노트북)]|
|ch06-03-hopt_nn.py      |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-03-hopt_nn.py) |[CODE(노트북)]|
|ch06-04-filter.py       |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-04-filter.py)|[CODE(노트북)]|
|ch06-05-embedded.py     |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-05-embedded.py)| [CODE(노트북)]|
|ch06-06-wrapper.py      |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch06/ch06-06-wrapper.py)| [CODE(노트북)]|

 ### ch07
   * 파이썬 소스 코드보기

|파일명|code(.py)|code(.ipynb)|
|------|---|---|
|ch07-01-stacking.py         |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch07/ch07-01-stacking.py)|[CODE(노트북)]|
|ch07-02-blending.py     |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch07/ch07-02-blending.py)|[CODE(노트북)]|
|ch07-03-adversarial.py     |[CODE(.py)](https://github.com/LDJWJ/kagglebook/blob/main/ch07/ch07-03-adversarial.py) |[CODE(노트북)]|



## 책에서의 관련 캐글 대회 및 정보 링크

#### ch01
  - 대회의 상위 솔루션 정리 [by sudalairajkumar](https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions)
  - 타이타닉 대회의 솔루션 정리 [by pliptor](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score)

#### ch02
 ### 평가지표로 보는 캐글 대회
   - Instacart Market Basket Analysis [대회로](https://www.kaggle.com/c/allstate-claims-severity)
    - 내용 : 보험 청구는 얼마나 심각한지에 대한 정도의 예측
    - 평가지표 : 평가지표(MAE)
 
  - Instacart Market Basket Analysis [대회로](https://www.kaggle.com/c/human-protein-atlas-image-classification/)
    - 내용 : Instacart 소비자는 어떤 제품을 다시 구매할까?
    - 평가지표 : 평가지표(mean F1 score)
    
  - Santander Product Recommendation [대회로](https://www.kaggle.com/c/santander-product-recommendation)
    - 내용 :Santander Bank  는 개인화 된 제품 추천 
    - 평가지표 : MAP@7 (Mean Average Precision @ 7)

  - Human Protein Atlas Image Classification [대회로](https://www.kaggle.com/c/hpa-single-cell-image-classification)
    - 내용 : 현미경 이미지에서 개별 인간 세포 차이 찾기 (다중 레이블 분류 문제)
    - 평가지표 : Macro F-Score

  - Quora Question Pairs 대회 [대회로](https://www.kaggle.com/c/quora-question-pairs)
    - 내용 : 같은 의도를 가진 질문 쌍을 식별하기
    - 평가지표 : 로그 손실

  - Home Credit Default Risk 대회 [대회로](https://www.kaggle.com/c/home-credit-default-risk)
    - 내용 : 각 신청자가 대출금을 상환 할 수있는 능력을 예측
    - 평가지표 : AUC

  - Two Sigma Connect: Rental Listing Inquiries [대회로](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries)
    - 내용 : RentHop의 새 임대 목록에 대한 관심은 얼마나됩니까?
    - 평가지표 : multi-class logloss
 [Link](https://medium.com/kaggle-blog/allstate-claims-severity-competition-2nd-place-winners-interviewalexey-noskov-f4e4ce18fcfc)
    - 클러스터 중심으로부터 거리를 특징으로 사용
  - 사이킷런의 cluster 모듈 [Link](https://scikit-learn.org/stable/modules/clustering.html)
  - 여러 변수를 조합한 지수 사용 - Recruit Restaurant Visitor Forecasting 대회 20th Solution [Link](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/49328)
  - Avito Demand Prediction Challenge 대회 9위 솔루션 (https://www.slideshare.net/JinZhan/kaggle-avito-demand-prediction-challenge-9th-place-solution-124500050)
     - 중요한 변수인 가격에 대해 상품명, 상품 분류(카테고리), 사용자나 지역 등 다양한 관점에서 평균과의 차와 비율 확인

