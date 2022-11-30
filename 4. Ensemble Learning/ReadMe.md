# Ensemble
앙상블은 하나의 알고리즘만 이용하는 것보다, 여러 개의 알고리즘의 결과를 종합할 때 더 나은 performance를 얻을 수 있다.  
데이터에 맞는 알고리즘을 찾기 위해서 일일이 trail & error를 하는 것은 부담스러운 작업이기 때문에 알고리즘을 결합하는 방식으로 준수한 performance를 낼 수 있다.  
- 앙상블의 목적: 다수의 모델을 학습하여 오류의 감소를 추구
  - 분산의 감소에 의한 오류 감소: 배깅(Bagging), 랜덤포레스트(Random Forest)
  - 편향의 감소에 의한 오류 감소: 부스팅(Boosting)
  - 분산과 편향의 동시 감소: Mixture of Experts
- 앙상블의 핵심
  - 다양성(diversity)을 어떻게 확보?
  - 최종 결과물을 어떻게 결합(combine, aggregate)할 것인가?
  
## Bias & Variance

<center>
  
$$y={ F }^{ * }(X)+\epsilon ,\quad \epsilon \sim N(0,{ \sigma  }^{ 2 })$$

</center>

$\F^*(x)$ 는 우리가 학습하고 싶은 데이터 생성 함수인데 노이즈때문에 정확한 추정은 현실적으로      불가능합니다. 동일한 데이터 생성 함수를 통해서 생성되었더라도 노이즈가 다르게 발생할 수 있기 때문에 함수식 $F^*(x)$가 달라질 수 있습니다. 

# Bagging
Bagging은 Bootstrap Aggregating의 줄임말로 모델의 학습 데이터가 달라짐에 따라 결과값이 달라지는 문제(variance)를 완화하기 위해 샘플을 여러번 뽑아(Bootstrap) 각 모델을 학습시켜 결과물을 Aggregation하는 기법입니다. 

<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb4wG8O%2FbtqyfYW98AS%2FYZBtUJy3jZLyuik1R0aGNk%2Fimg.png" height=400"></p>

우선 데이터로부터 bootstrap을 합니다. 복원 랜덤 샘플링 boostrap을 한 데이터로 모델을 학습시킵니다. 그리고 학습된 모델의 결과를 aggregation하여 최종 결과값을 구합니다.

## Out of bag error(OOB Error)
bagging에 사용되는 각각의 base learner에 대한 error는 bootstrap에서 제외된 데이터를 사용하여 base learner를 test합니다. 예로 1~6까지 데이터가 존재하고 base learner에 사용되는 data로 1/3/5가 선택되었다면 나머지 2/4/6이 test하고 각각의 base learner의 error를 계산합니다. 그리고 전체 모델의 error는 이 error들의 평균을 내게 되는데 이 때 발생하는 error를 Out of bag error(OOB Error)라고 합니다. 

## Result Aggregation

마지막에 Aggregation하는 방법은 target변수의 성질에 따라 다릅니다. Target변수가 연속형 변수라면 prediction값의 평균을 사용하고 범주형 변수라면 각 model의 사후확률의 평균 또는 분류 결과의 투표방법을 통해 결합하게 됩니다. 
- Majority voting
- Weighted voting(weight=training accuracy of individual models)
- Weighted voting(weight=predicted probability for each class)
- Stacking


## Bagging의 장단점  

Tree의 depth가 커지면 bias는 감소하지만 variance가 증가합니다. 하지만 bagging은 각 tree들의 bias는 유지하되, variance를 감소시키도록 하는 것이기 때문에 학습데이터의 noise에 영향을 덜 받는 장점이 있습니다.   
하지만 동시에 tree의 직관적인 이해력을 해치기 때문에 모형 해석에 어려움이 발생합니다.

# Random Forest
Random Forest는 overfitting을 방지하기 위해 최적의 기준 변수를 랜덤 선택하는 머신러닝 기법입니다. 여러개의 Decision tree(의사결정나무)를 만들고 숲을 이룬다는 의미에서 Forest라고 불립니다. 

## Random Forest Procedure
- Training set에서 표본 크기가 n인 bootstrap 수행
<p align="center"><img src="./img/1.png" height=200></p>

- Bootstrap sample에 대해 Random forest tree 모형 제작
  - 전체 변수중에서 m개 변수를 랜덤하게 선택
  - 최적의 classifier 선정
  - classifier에 따라 두 개의 자식 node 생성
<p align="center"><img src="./img/2.png" height=300></p>

- Tree 들의 앙상블 학습 결과 출력

## Random Forest 장점
- Classification & Regression에서 모두 사용 가능
- Missing value를 다루기 쉬움
- 대용량 데이터 처리에 효과적
- 모델의 노이즈를 심화시키는 Overfitting 문제를 회피하여, 모델 정확도 향상
- Classification 모델에서 상대적으로 중요한 변수를 선정 및 ranking 가능

# Boosting
boosting은 머신러닝 앙상블 기법 중 하나로 sequential한 weak learner들을 여러개 결합하여 예측 혹은 분류 성능을 높이는 알고리즘입니다. 
여러 개의 알고리즘이 순차적으로 학습-예측하면서 이전에 학습한 알고리즘의 예측이 틀린 데이터를 올바르게 예측할 수 있도록 다음 알고리즘에 가중치를 부과하여 학습과 예측을 진행합니다. 

<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FAafD5%2FbtqBuvD4hVP%2FJBhk9f2BXhnkLYd976yNTk%2Fimg.png" height=300></p>

## AdaBoost
AdaBoost는 Adaptive Boosting의 줄임말로 관측치들에 가중치를 더하면서 수행됩니다. 분류하기 어려운 인스턴스들에는 가중치를 더하고 이미 잘 분류되어진 인스턴스는 가중치가 덜합니다. 즉, weak learner의 오류에 가중치를 더하면서 boosting을 수행하는 알고리즘입니다. 

<p align="center"><img src="https://miro.medium.com/max/748/1*qzIPSA-HQlefxxZnPlb-2w.png" height=300></p>

## Gradient Boosting Machine(GBM)
Gradient Boosting Machine은 AdaBoost처럼 앙상블에 이전까지의 오차를 보정하도록 예측기를 순차적으로 추가합니다. AdaBoost와는 달리 매 반복마다 샘플의 가중치를 조정하는 대신에 이전 예측기가 만든 Residual Error에 새로운 예측기를 학습시킵니다.  
경사하강법(Gradient Descent)기법을 사용하여 최적화된 결과를 얻는 알고리즘으로 Gradient가 현재까지 학습된 classifier의 약점을 알려주고 이후 모델이 그것을 중점적으로 보완하는 방식입니다. 

<p align="center"><img src="https://miro.medium.com/max/560/1*85QHtH-49U7ozPpmA5cAaw.png" ></p>


## XGBoost
XGBoost는 Extreme Gradient Boosting의 약자로 앞서 다룬 GBM(Gradient Boosting Machine)을 개선시켜 정형, tabular 데이터의 분류 예측에 많이 사용되는 모델입니다. 

병렬 처리로 학습, 분류 속도가 빠르기 때문에 GBM보다 속도가 빠르다는 장점이 있습니다. overfitting regularization이 불가능한 GBM과 달리 overfitting regularization이 가능하며 CART(Classification and Regression Tree)앙상블 모델을 사용하여 회귀와 분류 모두 가능합니다. 

## Light GBM

LightGBM은 XGBoost의 효율성 문제를 보완하여 나온 알고리즘입니다. LightGBM의 GBM은 gradient boosting model로 tree를 기반으로 하는 학습 알고리즘입니다. GBM과 마찬가지로 틀린부분에 가중치를 더하면서 진행됩니다. 

기존의 tree기반 알고리즘은 level wise(균형 트리 분할) 방식이었습니다. 즉, 최대한 균형이 잡힌 트리를 유지하면서 분할하기 때문에 tree의 깊이가 최소화될 수 있었습니다. 하지만 이는 균형을 맞추기 위한 시간이 필요하다는 것을 의미합니다. LightGBM의 leaf wise tree분할 방식은 tree의 균형을 고려하지 않고 최대 손실 값을 가지는 leaf node를 지속적을 분할하면서 tree의 깊이가 깊어지고 비대칭적인 tree가 생성됩니다. 이와 같이 최대 손실값을 가지는 leaf node를 반복 분할하는 방식은 level wise tree 분할 방식보다 예측 오류 손실을 최소화할 수 있습니다. 

<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcnSeIU%2Fbtq5fw8Rgj8%2FhQgXvMy52q6YqjdJ1kUD4k%2Fimg.png" height=400></p>

LightGBM은 학습하는데 걸리는 시간이 적으며 메모리 사용량이 적은 장점이 있지만 작은 dataset을 사용할 경우 과적합 가능성이 크다는 단점을 갖고 있습니다. 

## CatBoost

CatBoost는 XGBoost와 더불어 level wise로 트리를 만들어나갑니다. 

- Ordered Boosting 방식
기존 Boosting 모델들은 훈련 데이터를 대상으로 잔차를 계산했지만 CatBoost는 학습데이터의 일부만으로 잔차계산을 한 뒤 이 결과로 모델을 다시 만들게 됩니다. Catboost는 학습데이터의 일부를 선택할 때 일종의 순서를 가지고 선택하게 됩니다. 이 때 순서는 임의로 정하기 때문에 데이터를 random하게 섞어주는 과정이 포함되어 있습니다. 이는 Boosting 모델이 overfitting을 방지하기 위해 지니고 있는 기법이라고 볼 수 있습니다. 

- 범주형 변수 처리  
CatBoost는 범주형 변수를 효율적으로 처리하는 장점을 갖고 있습니다. 데이터셋의 클래스를 명확하게 구분할 수 있는 중복되는 범주형 변수가 2개이상 존재할 때 이를 하나의 변수로 통합해 처리하는 것을 자동으로 수행합니다.(Categorical Feature Combination) 
CatBoost는 위에서 언급된 Ordered Boosting개념을 섞어 Response Encoding을 진행합니다. 학습데이터 중 일부를 선택하여 학습시키는데, 이 때 일부를 선택하는 'order'는 시간 순서를 따릅니다. 만약 데이터에 시계열 정보가 없다면 임의로 시간을 정하여 순서를 정하게 됩니다.(Random Permutation)

- 수치형 변수 처리
수치형 변수는 다른 일반 트리모델과 동일하게 처리합니다. 분기가 발생하면 gain, 즉 정보의 획득량이 높은 방향대로 나뉘게 됩니다. 수치형 변수가 많게 되면 LightGBM 처럼 시간이 오래 걸립니다. 따라서 범주형 변수를 다룰 경우에 Catboost를 사용하는 것이 좋습니다.

- 장점
Catboost는 시계열 데이터를 효율적으로 처리합니다. 속도가 빠르기 때문에 서비스에 예측 기능을 사용할 때 효과적일 수 있습니다. imbalanced dataset도 class_weight 파라미터 조정을 통해 예측력을 높일 수 있습니다. 
