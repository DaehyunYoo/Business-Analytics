# Semi-Supervised 이란?
적은 labeled data가 있으면서 추가로 활용할 수 있는 대용량의 unlabeled data가 있다면 semi-supervised learning을 고려할 수 있다. Semi-supervised learning은 소량의 labeled data에는 supervised learning을 적용하고 대용량 unlabeled data에는 unsupervised learning을 적용해 추가적인 성능향상을 목표로하는 방법론이다. 이런 방법론에 내재되는 믿음은 label을 맞추는 모델에서 벗어나 데이터 자체의 본질적인 특성이 모델링 된다면 소량의 labeled data를 통한 약간의 가이드로 일반화 성능을 끌어올릴 수 있다. 

<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FOaWXm%2FbtreCCEt8dI%2F14gNHsxjmFgOtONe7ejNjK%2Fimg.png" height=200></p>

Semi-supervised learning의 목적함수는 supervised loss $L_s$와 unsupervised loss $L_u$의 합을 최소화하는 것으로 표현할 수 있다. 즉, supervised, unsupervised를 1-stage로 한큐에 학습한다.

$$Loss = L_s + L_u $$

# Consistency Regularization(Consistency training)
이 방법은 unlabeled data point에 작은 perturbation을 주어도 예측의 결과에는 일관성이 있을 것이다라는 가정에서 출발한다. unlabeled data는 예측결과를 알 수 없기 때문에 data augmentation을 통해 class가 바뀌지 않을 정도의 변화를 주었을 때 원 데이터와 예측결과가 같아지도록 unsupervised loss를 주어 학습한다. 이를 통해 모델을 robust하게 할 수 있다.  
성능이 좋은 Semi supervised learning 모델은 대체로 consistency regualrization을 사용한다. 대표적인 모델로는 $\Pi$-model, Temporal Ensemble, Mean Teacher, Virtual Adversarial Training(VAT), unsupervised data augmentation(UDA)이 있다. 

## UDA(Unsupervised Data Augmentation)

<p align="center"><img src="https://joungheekim.github.io/img/in-post/2020/2020-12-13/overview.png" height=300></p>

### UDA Procedure

1. labeled data에 대해 통상의 방법으로 Cross Entropy를 최소화하는 방식으로 학습한다. 현 스텝의 분류 모델 $\theta$에 labeled input인 x를 넣어 그 출력이 label인 $y^*$와 최대한 유사해지도록 모델을 업데이트 한다. 
2. unlabeled data에 대해 Consistency loss를 최소화하는 방식으로 학습한다. 
- $p_\theta(y|x)$: 현 스탭의 분류 모델 $\theta$에 labeled input인 x를 넣어 범주확률을 계산
- $p_\theta(y|\hat x)$: 현 스탭의 분류 모델 $\theta$에 unlabeled input x으로부터 augmentation된 $\hat x$을 넣어 범주 확률 계산
- 위의 두 확률분포 사이의 KL-Divergence를 계산하는 것이 Consistency Loss
  
### UDA Contributution
- Data augmentation을 통한 Consistency training: unlabeled data에 대해 Consistency loss를 최소화하는 과정에서 노이즈에 강건한 모델 생성
- Semi-Supervised learning: Supervised Cross Entropy Loss와 Unsupervised Consistency Loss를 동시에 최소화하는 과정에서 labeled data의 정보가 unlabeled data 전파되도록 모델 업데이트

[UDA(Unsupervised Data Augmentation) Code]()
