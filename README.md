# Business-Analytics
이 내용은 강필성 교수님의 비즈니스 애널리틱스(2022년 2학기) 수업 내용을 참고한 내용입니다. 

## Dimensionality Reduction Overview

### Curse of Demensionality
실제 데이터의 차원은 매우 크더라도 내재된 차원(intrinsic/embedded dimension)은 원래 차원의 수보다 낮은 경우가 대부분

- 데이터의 차원이 높아짐에 따라 생기는 문제점
  - 변수가 증가할수록 잡음(noise)이 포함될 확률이 높아 예측 모델의 성능을 저하시킴
  - 변수가 증가할수록 예측 모델의 학습과 인식 속도가 느려짐
  - 변수가 증가할수록 예측 모델에 필요한 학습 집합의 크기가 커짐
  
- 차원의 저주를 극복하는 방법
  - 사전지식의 활용
  - 목적 함수의 smoothness를 증가시킴
  - 정량적 분석을 통한 차원 감소

### Dimensionality Reduction

- 차원축소: 배경
  - 이론적으로는 변수의 수가 증가할수록 모델의 성능이 향상됨(변수간 독립성 만족시)
  - 실제 상황에서는 변수간 독립성 가정 위배, 노이즈 존재 등으로 인해 변수의 수가 일정 수준 이상 증가하면 모델의 성능이 저하되는 경향이 있음
  
- 차원축소: 목적
  - 향후 분석 과정에서 성능을 저하시키지 않는 최소한의 변수 집합 판별
  
- 차원축소: 효과
  - 변수간 상관성을 제거하여 결과의 통계적 유의성 제고
  - 사후 처리(post-processing)의 단순화
  - 주요 정보를 보존한 상태에서 중복되거나 불필요한 정보만 제거
  - 고차원의 정보를 저차원으로 축소하여 시각화(visualization) 가능
 
- 차원축소 방식
  - Supervised dimensionality reduction
    - 축소된 차원의 적합성을 검증하는데 있어 예측 모델을 적용
    - 동일한 데이터라도 적용되는 예측 모델에 따라 축소된 차원의 결과가 달라질 수 있음
  
  - Unsupervised dimensionality reduction
    - 축소된 차원의 적합성을 검증하는데 있어 예측 모델을 적용하지 않음
    - 특정 기법에 따른 차원축소 결과는 동일
 
- 차원축소 기법
  - 변수 선택(variable/feature selection)
    - 원래의 변수 집단으로부터 유요알 것으로 판단되는 소수의 변수들을 선택
    - Filter - 변수 선택 과정과 모델 구축 과정이 독립적
    - Wrapper - 변수 선택 과정이 머신러닝 모델의 결과를 최적화 하는 방향으로 이루어짐
    
  - 변수 추출(variable/feature extraction)
    - 원래의 변수 집단을 보다 효율적인 적은 수의 새로운 변수 집단으로 변환하는 것
    - 머신러닝 모델에 독립적인 성능 지표가 추출된 변수의 효과를 측정하는데 사용됨
    
![image](https://user-images.githubusercontent.com/112569789/194992865-312ada9a-68e0-4364-b64c-f4da0e9ac5a7.png)

-----------------------------------------------------------------------------------------------------------------

## Dimensionality Reduction
1. Supervised Method
- [Genetic_algorithm]#

2. Unsupervised Method - Linear
- [MDS](https://github.com/YooD11/Business-Analytics/blob/main/MDS.ipynb)

3. Unsupervisd Method - Nonlinear
- [Unsupervised Method(Nonlinear)](https://github.com/YooD11/Business-Analytics/blob/main/Unsupervised_nonlinear.ipynb) 
 
    
 
 
 
