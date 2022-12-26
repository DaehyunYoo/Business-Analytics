# Related Work

## Consistency Regularization
Consistency는 input data를 augmentation한 것에 대해 prediction의 일관성을 의미합니다. 즉, 약간 변형한 데이터를 넣어도 일관성있는 예측값을 얻을 수 있도록 하는 것입니다. 

<p align="center"><img src="https://blog.kakaocdn.net/dn/epU6yc/btqO0Tj2N4G/PIdRbnzWiD9096CUGrKVgk/img.png"></p>

수식에서 Augment()는 stochastic data augmentation을 의미합니다. 수식을 통해 모델은 augmented된 data에 대해 같은 class로 분류되도록 학습된다는 것을 알 수 있습니다.

## Entropy Minimization
Entropy Minimization은 모델 classifier가 unlabeled data의 예측 entropy를 최소화하는 것입니다. 이는 unlabeled data에 대한 예측을 확신하도록 해당 class에 대한 prediction probability를 높이도록 학습함을 의미합니다.  


Entropy Minimization의 대표적방법은 Pseudo-Labeling입니다. 이는 unlabeled data에 대해 confidence를 높여 implicit하게 예측하고 이 예측한 label을 출력하는 것입니다.

<p align="center"><img src="https://www.researchgate.net/publication/345432485/figure/fig1/AS:955199128608771@1604748627675/Pseudo-Labeling-Learning-Architecture.png" height=250></p>

## Mixup 

초록색을 Class1, 주황색을 Class2, 파란색은 prediction probability라고 하고 class1, class2의 비율이 0.8, 0.2 비율이 되도록 섞습니다. 그럼 데이터가 변형되고 모델은 새로운 데이터라고 인식하고 학습하게 됩니다. 이렇게 가짜 label이 형성되면서 perturbation에 대해 강건한(robust)하게 학습할 수 있습니다.

<p align="center"><img src="https://euphoria0-0.github.io/assets/img/posts/2021-01-08-Semi-Supervised-Learning-and-MixMatch/MixUp.png" height=250></p>


# MixMatch 
MixMatch는 앞선 related work를 합친 모델입니다. 

<p align="center"><img src="https://blog.kakaocdn.net/dn/by9DMc/btrbABCX4gg/sUexA73kRKYGHbtRmFk3HK/img.png"></p>

## Method
### 1. Data Augmentation
먼저 labeled data $x_b$와 unlabeled data $u_b$를 augmentation 합니다. 

$$\hat{x}_b = \textrm{Augment}(x_b) \\ \hat{u}_{b,k} = \textrm{Augment}(u_b)$$ 

### 2. Label Guessing
Augmentation된 data를 이용해 분류, 예측을 실시합니다. 이렇게 예측한 클래스 레이블을 guessed label $q_b$라고 합니다.

### 3. Sharpening
이제 이에 대해 Entropy Minimization을 하기 위해 sharpening을 합니다. Augmentation된 unlabeled data로도 분류를 잘하기 위해 하나의 예측된 클래스의 확률을 가장 높이고 나머지 다른 클래스에 대한 확률을 줄임으로써 예측에 대한 불확실성 즉, entropy를 줄입니다.

### 4. Mixup
위에서 augmented되고 entropy가 minimize된 labeled data와 unlabeled data를 섞습니다. 

### 5. Prediction
학습을 하기 위한 loss function을 정의합니다. 

$$\mathcal{L}_{\mathcal{X}}=\frac{1}{|\mathcal{X}'|}\sum_{x,p \in \mathcal{X}'}H(p,p_{\textrm{model}}(y|x;\theta)) $$

$$\mathcal{L}_{\mathcal{U}}=\frac{1}{L|\mathcal{U}'|}\sum_{u,q \in \mathcal{U}'}\|q-p_{\textrm{model}}(y|u;\theta)\|_2^2$$

$$\mathcal{L}=\mathcal{L}_{\mathcal{X}}+\lambda_{\mathcal{U}}\mathcal{L}_{\mathcal{U}} $$

$\mathcal{L}_\mathcal{X}$는 augmented labeled data에 대한 training loss 입니다. 다음 $\mathcal{L}_\mathcal{U}$는 augmented unlabeled data에 대한 loss입니다. Consistency Regularization에 의해 나타나며, 모델이 augmentation한 같은 데이터들에 대해 일관된 예측을 하는지 보기 위한 loss입니다. 이는 unlabeled data에 대한 loss이자 predictive uncertainity에 대한 measure로 해석할 수 있습니다.

# FixMatch
## Method

<p align="center"><img src="https://blog.kakaocdn.net/dn/DiVQc/btqO0fASFqk/BeatuIw8TOEq0fYxZNSRT0/img.png" height=250></p>

Fixmatch에서는 Strong augmentation와 Weak augmentation가 적용됩니다. Weak augmented 된 이미지를 사용해 pseudo label을 만듭니다. Weak augmentation은 기본적인 flip, rotation등의 augmentation을 사용하였으며, strong augmentation은 사람도 잘 알아보기 어렵게 강한 augmentation을 적용한 것입니다. 

Augmentation된 두 이미지를 동일한 model에 forward시킵니다. 그러면 softmax를 통과한 prediction value 값이 output으로 나오고, Weak augmentation을 통과한 output에서 가장 probability가 높은 값의 class를 Pseudo-Labeling해줍니다. 기존의 UDA(Unsupervised Data Augmentation)이나 consistency에서 사용했던 sharpening 기법과 동일하게 entropy를 낮춰주는 행위와 동일합니다. 그렇게 만든 Pseudo-Label을 strong augmentation의 output으로 나온 값과 CrossEntropy연산을 하여 loss term을 둔 것이 FixMatch입니다. 

- Supervised Loss function

$$l_s\ =\ \frac{1}{B}\sum _{b=1}^BH\left(p_b,\ p_m\left({y}|{a\left(x_b\right)}\right)\right)
$$

- Unsupervised Loss function

$$l_u\ =\ \frac{1}{\mu B}\sum _{b=1}^{\mu B}1\left(\max _{ }^{ }\left(q_b\right)\ge T \right)\ H\left(\hat{q_b},\ p_m\left({y}|{A\left(x_b\right)}\right)\right)
$$


# FlexMatch
## Method

FlexMatch는 FixMatch의 단점을 보완한 모델입니다. FixMatch는 해당 Class에 속할 확률(Softmax)값이 일정 Threshold를 넘을때 해당 Class에 속할 확률을 1로 pseudo-labeling하고, 넘지 못하면 해당 샘플은 학습에 참여하지 못합니다. 때문에 제거되는 샘플이 많아질 수 있는 단점이 존재합니다. 따라서 FlexMatch는 이런 문제를 해결하고자 Class별로 confidence에 따라 다른 Threshold값을 적용하는 방법을 사용하였습니다. 

FlexMatch는 (CPL)Curriculum pseudo labeling을 통해 각 class의 confidence에 따른 Confidence Threshold를 다르게 조정합니다. 

<p align="center"><img src="https://storrs.io/content/images/size/w1000/2021/11/image2--12-.png"></p>

첫번째로 confidence(예측 확률값의 max)가 미리 지정한 threshold를 넘기면서 해당 class로 예측된 unlabeled sample의 개수를 사용하여 Learning effect를 측정합니다.
두번째는 0~1 사이값으로 learning effect를 정규화합니다. 마지막으로 세번째 class별 threshold를 업데이트하고 threshold보다 높으면 pseudo-label로 활용하여 loss를 계산하고 threshold보다 낮으면 loss=0으로 설정합니다.

<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/d85de62abb78b184d02e4c9761b98fa9e7dca6ca/3-Figure1-1.png" height=200></p>
