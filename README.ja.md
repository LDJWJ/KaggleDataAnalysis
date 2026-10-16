🇰🇷 [한국어](README.md) · 🇺🇸 [English](README.en.md) · 🇯🇵 **日本語**

# KaggleDataAnalysis

> **補足：** 教材（PDF/HTML）はすべて韓国語です。コード自体は言語に依存せずそのまま動作し、HTMLページはブラウザ（Chromeなど）の自動翻訳で読めます。

## Titanic 
  * タイタニックの超かんたんEDAとモデル構築 
    [html](https://ldjwj.github.io/KaggleDataAnalysis/Titanic/01_Baselinemodel-onkaggle_v02_504.html)
    [pdf](./Titanic/01_Baselinemodel-onkaggle_v02_504.pdf)

  * タイタニックのデータ分割とモデル比較  
    [html](https://ldjwj.github.io/KaggleDataAnalysis/Titanic/01B_modellevelup_onkaggle_v03_504.html)
    [pdf](./Titanic/01B_modellevelup_onkaggle_v03_504.pdf)

  * データ可視化、データ前処理、モデル改善 
    [html](https://ldjwj.github.io/KaggleDataAnalysis/Titanic/02_DATA_EDA_V11_504.html)
    [pdf](./Titanic/02_DATA_EDA_V11_504.pdf)

  * 特徴量エンジニアリング、ハイパーパラメータチューニングなどによるモデル改善 
    [html](https://ldjwj.github.io/KaggleDataAnalysis/Titanic/03_Model_LevelUp_V10.html)
    [pdf](./Titanic/03_Model_LevelUp_V10.pdf)

  * Titanicモデル改善ソリューションのまとめ
    * https://www.kaggle.com/code/pliptor/how-am-i-doing-with-my-score

## Bike Sharing Demand 
  * EDAとモデル構築 
    [html](https://ldjwj.github.io/KaggleDataAnalysis/Bike/01_Bike_First_Model_V11.html)
    [pdf](./Bike/01_Bike_First_Model_V11.pdf)

  * データ可視化、データ前処理、モデル改善 
    [html](https://ldjwj.github.io/KaggleDataAnalysis/Bike/02_BikeSharingDemand_EDA_V11.html)
    [pdf](./Bike/02_BikeSharingDemand_EDA_V11.pdf)

  * 特徴量エンジニアリング、ハイパーパラメータチューニングなどによるモデル改善 
    [html](https://ldjwj.github.io/KaggleDataAnalysis/Bike/03_BikeSharingDemand_MultiModel_V11.html)
    [pdf](./Bike/03_BikeSharingDemand_MultiModel_V11.pdf)



## 関連リンクと資料 
### Kaggler
* 圧倒的な強さと洗練された解法でKagglerたちに衝撃を与え、2018年にKaggleランキング1位を獲得
  * [bestfitting](https://medium.com/kaggle-blog/profiling-top-kagglers-bestfitting-currently-1-in-the-world-58cc0e187b)

* [Winning Data Science Competitions](https://www.slideshare.net/OwenZhang2/tips-for-data-science-competitions)
2015年まで長くKaggleランキング1位だったOwenのスライド

* [Kaggle Grandmaster Kohei Blog](https://i.ho.lc/pages/about.html) https://i.ho.lc/pages/about.html
* Kaggleコンペ Crowdflower Search Results Relevance の1位ソリューション [Link](https://github.com/ChenglongChen/kaggle-CrowdFlower/blob/master/Doc/Kaggle_CrowdFlower_ChenglongChen.pdf)

### Kaggle上位解法/インタビュー
 * https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions
 * Yelp Restaurant Photo Classification - [1st place Dmitrii Tsybulevskii](https://engineeringblog.yelp.com/2016/04/yelp-kaggle-photo-challenge-interview-1.html)
 * Instacart Market Basket Analysis, [2nd place solution](https://medium.com/kaggle-blog/instacart-market-basket-analysis-feda2700cded) [2nd place Github](https://github.com/KazukiOnodera/Instacart)
 * Avito Demand Prediction Challenge [9th place](https://www.slideshare.net/JinZhan/kaggle-avito-demand-prediction-challenge-9th-place-solution-124500050)

### 役立つドキュメント
 * [How to use Kaggle](https://www.kaggle.com/docs/competitions)
 * [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html) scikit-learnのドキュメント
 * FEATURE ENGINEERING HJ van Veenによる特徴量エンジニアリングの資料 [Link](https://www.slideshare.net/HJvanVeen/feature-engineering-72376750)
 - How to explain gradient boosting [Terence Parr and Jeremy Howard](https://explained.ai/gradient-boosting/index.html)


### 役立つ講座
 * CourseraのKaggle講座
 * [Coursera - How to Win a Data Science Competition:Learn from Top Kagglers](https://www.coursera.org/learn/competitive-data-science/)

### ライブラリと関連論文
 * xgboost
   * https://xgboost.readthedocs.io/en/latest/
   * [Github](https://github.com/dmlc/xgboost/)
 * lightgbm
   * https://lightgbm.readthedocs.io/en/latest/
   * [Github](https://github.com/microsoft/LightGBM/)
 * catboost
   * https://catboost.ai/docs/
   * [Github](https://github.com/catboost/catboost)
 * keras
   * https://keras.io/
   * [Github](https://github.com/keras-team/keras)
 * pytorch
   * https://pytorch.org/
   * [Github](https://github.com/pytorch/pytorch)
 * tensorflow
   * https://www.tensorflow.org/
   * [Github](https://github.com/tensorflow/tensorflow)
 * hyperopt
   * https://github.com/hyperopt/hyperopt/wiki
   * [Github](https://github.com/hyperopt/hyperopt)
 * optuna
   * https://optuna.readthedocs.io/en/latest/
   * [Github](https://github.com/pfnet/optuna)
 * パラメータチューニング
   * [lightgbm] https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
   * [catboost] https://catboost.ai/docs/concepts/parameter-tuning.html
   * [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
   * [PARAMETERS](https://sites.google.com/view/lauraepp/parameters)
 * [日本語] Hyperoptについて
   * [Link](https://www.slideshare.net/hskksk/hyperopt)
     
 ### 分析事例
   * Kaggle Home Depot Product Search Relevance - Turing Test's 3rd Place Solution
     * [Link](https://github.com/ChenglongChen/Kaggle_HomeDepot/blob/master/Doc/Kaggle_HomeDepot_Turing_Test.pdf)
     * [Github[(https://github.com/ChenglongChen/Kaggle_HomeDepot)

 
 ### Kaggleコンペ一覧
*  Titanic: Machine Learning from Disaster
[Link](https://www.kaggle.com/c/titanic)
 *  House Prices: Advanced Regression Techniques
[Link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
 *  Heritage Health Prize
[Link](https://www.kaggle.com/c/hhp)

 *  Display Advertising Challenge
[Link](https://www.kaggle.com/c/criteo-display-ad-challenge)
 *  Microsoft Malware Classification Challenge (BIG 2015)
[Link](https://www.kaggle.com/c/malware-classification)
 *  Otto Group Product Classification Challenge
[Link](https://www.kaggle.com/c/otto-group-product-classification-challenge)
 *  Walmart Recruiting II: Sales in Stormy Weather
[Link](https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather)

 *  Facebook Recruiting IV: Human or Robot?
[Link](https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot)
 *  Crowdflower Search Results Relevance
[Link](https://www.kaggle.com/c/crowdflower-search-relevance)
 *  Caterpillar Tube Pricing
[Link](https://www.kaggle.com/c/caterpillar-tube-pricing)
 *  Coupon Purchase Prediction
[Link](https://www.kaggle.com/c/coupon-purchase-prediction)

 *  Rossmann Store Sales
[Link](https://www.kaggle.com/c/rossmann-store-sales)
 *  Walmart Recruiting: Trip Type Classification
[Link](https://www.kaggle.com/c/walmart-recruiting-trip-type-classification)
 *  Airbnb New User Bookings
[Link](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings)

 *  Prudential Life Insurance Assessment
[Link](https://www.kaggle.com/c/prudential-life-insurance-assessment)
 *  BNP Paribas Cardif Claims Management
[Link](https://www.kaggle.com/c/bnp-paribas-cardif-claims-management)
 *  Home Depot Product Search Relevance
[Link](https://www.kaggle.com/c/home-depot-product-search-relevance)
 *  Santander Customer Satisfaction
[Link](https://www.kaggle.com/c/santander-customer-satisfaction)
 *  Bosch Production Line Performance
[Link](https://www.kaggle.com/c/bosch-production-line-performance)

 *  Allstate Claims Severity
[Link](https://www.kaggle.com/c/allstate-claims-severity)
 *  Santander Product Recommendation
[Link](https://www.kaggle.com/c/santander-product-recommendation)
 *  Two Sigma Financial Modeling Challenge
[Link](https://www.kaggle.com/c/two-sigma-financial-modeling)
 *  Data Science Bowl 2017
[Link](https://www.kaggle.com/c/data-science-bowl-2017)
 *  Two Sigma Connect: Rental Listing Inquiries
[Link](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries)

 *  Quora Question Pairs
[Link](https://www.kaggle.com/c/quora-question-pairs)
 *  Mercedes-Benz Greener Manufacturing
[Link](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing)
 *  Instacart Market Basket Analysis
[Link](https://www.kaggle.com/c/instacart-market-basket-analysis)
 *  Web Traffic Time Series Forecasting
[Link](https://www.kaggle.com/c/web-traffic-time-series-forecasting)
 *  Text Normalization Challenge - English Language
[Link](https://www.kaggle.com/c/text-normalization-challenge-english-language)
