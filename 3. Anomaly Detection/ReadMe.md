# Anomaly Detection
## Novel Data
Novel Data란 일반적인 데이터의 경향과는 다른 특이한 데이터를 뜻합니다. 정상적인 데이터를 생성하는 매커니즘을 위반하여 생성되며 Noise Data와는 다릅니다. Noise Data는 측정과정에서의 무작위성에 기반하며 Noise만을 제거하는 것은 현실적으로 불가능합니다. 
Anomaly Data는 Novel Data와 마찬가지로 이상치 데이터를 뜻합니다. 하지만 Anomaly Data는 조금 더 부정적인 의미에 가깝고 Novelty는 긍정에 가깝다는 차이입니다. 

## Anomaly Detection vs Classification


<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTU3/MDAxNjY4MzkzNzk4ODA1.JPnA7MRzlMmtkATC3zU-zWUT9lcwzij0SOuQ-uo2GpQg.Hs1zLo4AOfolHvyzFGJ78SoIlon3Rt4XICoj6AWeBawg.PNG.dhyoo9701/1.png?type=w773" height=200"></p>

Anomaly Detection은 한쪽 범주만 학습하여 정상 범주에 포함되지 않으면 이상치로 분류하고 Classification은 양쪽 범주를 모두 학습합니다. 그런데 이 기준은 보통 범주간 불균형이 심한 약 0.1:99.9정도로 심한 경우나, 수가 적은 클래스가 충분하지 않다면 Anomaly Detection에 해당됩니다. 반대로 class imbalance가 8:2, 7:3이나 애매한 정도라면 class imbalance를 해결할 수 있는 방법을 찾고 classification에 해당됩니다. Classification의 성능이 보다 우수할 가능성이 높습니다. 

# Density-based Anomaly Detection
Density-based Anomaly Detecion 밀도기반 추정법은 주어진 데이터를 바탕으로 각 객체들이 생성될 확률을 추정하는 것입니다. 새로운 데이터가 생성될 확률이 낮을 경우 이상치로 판단하고 정상 데이터로 분포를 만들어놓고 새로운 Data를 추가해 위치를 보고 이상치인지 아닌지를 판별하게 됩니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTU4/MDAxNjY4MzkzNzk4ODAz.7BDWooKOL6wFiXsvdJ1XHRZd_ZiFNHKK57mIC9d5P6wg.rSXr6ixP2E9-anVeQ-8Fe_A9LiCTmB3M48nP8e1SGe4g.PNG.dhyoo9701/2.png?type=w773" height=200"></p>
  

밀도 기반 추정법은 일반적으로 Gaussian density estimation, Mixture of Gaussian estimation, Kernel density estimation, Parzen window density estimation, Local Outlier factors가 있습니다. 

## 1.Gaussian  Density Estimation
Gaussian Density Estimation은 모든 데이터가 하나의 Gaussian(정규)분포로 부터 생성됨을 가정합니다.  
학습과정에서는 주어진 정상 데이터들을 통해 Gaussian Distribution의 평균 벡터와 공분산 행렬을 추정합니다. 테스트 과정에서는 새로운 데이터에 대하여 생성 확률을 구하고 이 확률이 낮을수록 이상치에 가까운 것으로 판정합니다.  



<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfNjMg/MDAxNjY4MzkzNzk4ODA3.6-GosS-N4Qlhl0oxUj41_8g0B7rOGe2zvMA5nTuR3cIg.nsPXX1TRdK_Jb8nI4EGeyDWkYDF3JdtF4_OjWc0qy7Ig.PNG.dhyoo9701/3.png?type=w773" height=200></p>
</center>

<center>
$$p(x)\quad =\quad \frac { 1 }{ { 2\pi  }^{ { d }/{ 2 } }{ \Sigma  }^{ { 1 }/{ 2 } } } exp\left[ \frac { 1 }{ 2 } { (x-\mu ) }^{ T }{ \Sigma  }^{ -1 }(x-\mu ) \right]$$

$$\mu \quad =\quad \frac { 1 }{ n } \sum _{ { x }_{ i }\in { X }^{ + } }^{  }{ { x }_{ i } } \quad (mean\quad vector)$$

$$\Sigma \quad =\quad \frac { 1 }{ n } \sum _{ { x }_{ i }\in { X }^{ + } }^{  }{ ({ x }_{ i }-\mu ){ ({ x }_{ i }-\mu ) }^{ T } } \quad (covariance\quad matrix)$$
</center>

Gaussian 분포는 Covariance type에 따라 모양이 변합니다. 

- Spherical type

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjYz/MDAxNjY4MzkzNzk4ODYx.TWTdydUMo7YOShiRllWyOY5qP-vZ5Nw-OFW_2hMTZxsg.YwXaCjnOyE93iehy2GGGYDxYHxFoeTtMyTp86ONwu-0g.PNG.dhyoo9701/4.png?type=w773" height=200></p>

covariance matrix가 diagonal이고 동시에 x1과 x2의 각 축에서 분산이 같다고 가정할 때 Gaussian 분포는 위에서 봤을 때 원모양의 등고선을 나타냅니다. 

- Diagonal

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjE0/MDAxNjY4MzkzNzk4ODcx.63zM0-kdt6yCIzp7jWeOKwUl9FOfOcI2IR-WXLQI4rcg.9R1Nd5jPor9eaSmWO6s99D3RdbO9LYAq2B55O_8zCcAg.PNG.dhyoo9701/5.png?type=w773" height=200></p>


covariance matrix가 diagonal이지만 축마다 분산이 다르다고 가정한 경우의 Gaussian분포는 위에서 봤을 때 축이 어그러지지 않은 타원형의 등고선을 나타냅니다. 

- Full
  
<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjIy/MDAxNjY4MzkzNzk5MDI3.LS4PmDosTpv7b-WSZVz5vvnp6R9075flrEGnHanxwrcg.ViRi1pWjJ-cVsVYvlCl4AhTSh7csa1BAZuvs1kbwYwEg.PNG.dhyoo9701/6.png?type=w773" height=200></p>

Full covariance matrix를 가질 때를 가정하면 Gaussian분포가 축도 어그러진 타원형의 등고선 형태를 보입니다. 

## 2. Mixture of Gaussian
Gaussian Density Estimation은 데이터의 분포가 Gaussian 분포를 따른다는 강한 가정을 하고 있기 댸문에 복잡한 데이터의 분포는 잘 표현하지 못할 수 있습니다. 이런 이유 때문에 여러 개의 Gaussian 분포의 결합으로 데이터를 나타내고자 MoG(Mixture of Gaussian)을 사용합니다. 

MoG는 데이터는 여러 개의 Gaussian 분포의 혼합으로 이루어져 있음을 허용하며 이 Gaussian 분포들의 선형 결합으로 전체데이터의 분포를 표현하게 됩니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjAw/MDAxNjY4MzkzNzk5MDI3.eZBy4pxjjXMGLKibzSyPHpT4kW21OL4WCZAnv-xqLX0g.ViRi1pWjJ-cVsVYvlCl4AhTSh7csa1BAZuvs1kbwYwEg.PNG.dhyoo9701/7.png?type=w773" height=200></p>

- MoG에서 데이터가 normal일 확률 

<center>
$$p(x|\lambda )=\sum _{ m=1 }^{ M }{ { w }_{ m }g(x|{ \mu  }_{ m },{\sum} _{ m }  } )$$
</center>

- MoG에서 각 Gaussian Distribution의 g와 lambda

<center>

$$g(x|{ \mu  }_{ m },{ \sum   }_{ m })=\frac { 1 }{ { (2\pi ) }^{ d/2 }{ |{ \sum   }_{ m }| }^{ 1/2 } } exp[\frac { 1 }{ 2 } (x-{ \mu  }_{ m })^{ T }{ { \sum   } }_{ m }^{ -1 }(x-{ \mu  }_{ m })]$$

</center>  

<center>
  
$$\lambda = \lbrace{w_m, \mu_m,\sum_m}\rbrace, m=1, \cdot \cdot \cdot, M$$

</center>

### EM 알고리즘 

MoG(Mixture of Gaussian)모델을 사용하기 위해서는 EM Algorithm(Expectation-Maximization Algorithm)을 사용해야 합니다. EM Algorithm은 기대값 최대화 알고리즘으로 latent variable을 도입하여 최대 우도 추정량을 구하는 방법입니다. E(Expectation)단계에서는 로그 우도의 기대값을 계산하고, M(Maximization)단계는 기댓값을 최대화하는 단계로 E,M 단계를 반복 수행합니다. 

- Expectation

<center>

$$p(m|{ x }_{ i },\lambda )=\frac { { w }_{ m }g({ x }_{ t }|{ \mu  }_{ m },{ m }_{ m }) }{ \sum _{ k=1 }^{ M }{ { w }_{ k }g({ x }_{ i }|{ \mu  }_{ k },{ m }_{ k }) }  }$$

</center>

- Maximization

<center>

$${ w }_{ m }^{ (new) }=\frac { 1 }{ N } \sum _{ i=1 }^{ N }{ p(m|{ x }_{ i },\lambda ) }$$

$${ \mu  }_{ m }^{ (new) }=\frac { p(m|{ x }_{ i },\lambda ){ x }_{ i } }{ \sum _{ i=1 }^{ N }{ p(m|{ x }_{ i },\lambda ) }  }$$

$${ \sigma  }_{ m }^{ 2(new) }=\frac { \sum _{ i=1 }^{ N }{ p(m|{ x }_{ i },\lambda ){ x }_{ i }^{ 2 } }  }{ \sum _{ i=1 }^{ N }{ p(m|{ x }_{ i },\lambda ) }  } -{ \mu  }_{ m }^{ 2(new) }$$

</center>

- 데이터의 분포를 추정하는 과정

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTIx/MDAxNjY4MzkzNzk5MDYw.5zWtUMJLqXRYIr4vL0OIV8Mtm0pp66z3goUvgEPlf2og.GZ3Hq4eCY-iVUBK7qYyvksn8FQuJVczqptWGR3fAF3Yg.PNG.dhyoo9701/8.png?type=w773" height=200></p>

MoG(Mixture of Gaussian) 역시 Covariance type에 따라 모양이 변합니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjE0/MDAxNjY4MzkzNzk5MTQ5.wdUdDOACcEqJl6WEOR_c2iko1tbRy4tUPYtLAsjiMyYg.a9yxIvRp9MyKvB7zISPUHh1rba85mEYTLh0mb42Cqmwg.PNG.dhyoo9701/10.png?type=w773" height=300></p>

## 3. Kernel-density Estimation
Kernel density Estimation은 데이터가 Gaussian 분포와 같은 특정 분포를 따르지 않는다고 가정합니다. 즉, 데이터 자체에서 바로 밀도 추정을 하고자하는 방법입니다. 모수의 분포를 모를때 밀도를 추정하는 가장 단순한 방법은 히스토그램이지만, bin의 경계에서 불연속성이 나타나고, bin의 크기 및 시작위치에 따라 히스토그램이 달라지는 문제점이 존재합니다. 이러한 연속성의 문제를 해결하기 위한 방법이 Kernel Density Estimation입니다. 

임의의 확률밀도함수 p(x)의 분포에서 벡터x가 표본 공간에 주어진 R범위에 있을 확률은 다음과 같습니다. 

<center>

$$P=\int _{ R }^{  }{ p({ x }^{ \prime  })dx^{ \prime  } }$$

</center>

확률밀도의 추정은 샘플 수가 클수록 영역의 크기가 작을수록 정확합니다. 실제 상황에서는 데이터의 수가 고정되어 있으므로 적절한 영역을 찾는 문제로 귀결됩니다. 영역을 고정시키고 영역 안의 데이터 샘플을 찾는 것이 Kernel Density Estimation의 목적이라고 할 수 있습니다. 

<p align="center">
<img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjAx/MDAxNjY4MzkzNzk5MjE1.AfqzHeQn7N8BJovsLy2gE3dmgRS7z5ZDq1IQrrHeae4g.jnHn5o5CjQo0PyZ7LACYARxWL99jUvoQdCfHckUrFEIg.PNG.dhyoo9701/12.png?type=w773" height=300></p>

## 4. Parzen Window Density Estimation

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfNTAg/MDAxNjY4MzkzNzk5MTk1.VRV7pzjeR_Oaaq3f6lOjVotB6QE8B0zPCUlvlllTL-wg.Mb1qgxH_I3ey7somprf9PzgNPZyBQA7Pf72jQP-k3FIg.PNG.dhyoo9701/11.png?type=w773" height=300></p>

그림과 같이 k개의 샘플을 갖고 있는 hypercube의 영역 R이 있다고 한다면 차원 d의 부피는 $h_n^d$ 입니다. 

hypercube안의 샘플 개수를 세는 수식은 다음과 같습니다. k는 hypercube안의 샘플 개수를 나타내며, x를 기준으로 각 차원에서 h/2 거리 안에 있는 모든 샘플의 개수를 세고 있는 식입니다. 이러한 방식은 각 hypercube의 경계값 위에 데이터가 있을 때는 이를 처리하는 것이 모호하다는 단점이 있습니다. 이를 해결하기 위해 다른 커널 함수를 사용할 수 있습니다. 

<center>

$$% <![CDATA[
% <![CDATA[
K({ u })=\left\{ \begin{array}{lll} 1, & |u_{ i }|\le 1/2, & i=1,...,D \\ 0, & otherwise & \;  \end{array} \right. \qquad %]]> %]]>$$

$$k=\sum _{ i=1 }^{ N }{ K(\frac { { x }^{ i }-x }{ h } ) }$$

</center>

Parzen window를 통해 데이터의 밀도를 추정하는 함수는 다음과 같습니다. 

<center>

$$p(x)=\frac { 1 }{ N{ h }^{ d } } \sum _{ i=1 }^{ N }{ K(\frac { { x }^{ i }-x }{ h } ) }$$

</center>

### Smooth Kernel function
K(U)함수는 영역 안을 1 밖이면 무조건 0을 주기 때문에 가장자리 영역에서는 불연속적인 값을 갖는 단점이 있습니다. 즉, discrete한 밀도 표현이 문제가 되고 연속형 밀도 추정에 부적합하다는 뜻입니다. 그래서 사용한 기법이 Smooth기법입니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfNDcg/MDAxNjY4MzkzNzk5Mjcy.pCIVAq3yx6Nle6qqsLpqJnQa_wGNo0KsAGg0MszkEbMg.BE12wPxmKu_zI102avLL3lJQVzFZ72Kg8XkJK3oTEYog.PNG.dhyoo9701/13.png?type=w773" ></p>

다음 그림과 같이 표현하면 가장자리에서 연속적인 값도 표현할 수 있습니다. 예를 들어 Gaussian kernel을 사용하면 p(x)는 아래 수식과 같습니다. 관찰한 각 데이터 별로 하나의 Gaussian 분포가 만들어지고 이를 선형 결합하면 알고자하는 분포가 완성됩니다. 사용자가 이 kernel을 어떻게 지정하느냐에 따라 최종적으로 추정한 분포의 모습이 바뀌게 됩니다. 

Smooth parameter(bandwidth) h
- Large h: 밀도가 크고 완만한 분포로 추정
- Small h: 뾰족한 밀도의 분포로 추정

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjE0/MDAxNjY4MzkzNzk5MTQ5.wdUdDOACcEqJl6WEOR_c2iko1tbRy4tUPYtLAsjiMyYg.a9yxIvRp9MyKvB7zISPUHh1rba85mEYTLh0mb42Cqmwg.PNG.dhyoo9701/10.png?type=w773" height=300></p>

## 5. Local Outlier Factors(LOF)
LOF(Local Outlier Factors)는 데이터가 가지는 상대적인 밀도까지 고려한 이상치 탐지방법입니다. 각각의 관측치가 데이터 안에서 얼마나 벗어나 있는가에 대한 정도(이상치 정도)를 나타냅니다. LOF의 가장 중요한 특징은 모든 데이터를 전체적으로 고려하는 것이 아니라, 해당 관측치의 주변 데이터(neighbor)를 이용하여 local관점으로 이상치 정도를 파악하는 것입니다. 또한 주변 데이터를 몇개까지 고려할 것인가를 나타내는 k라는 hyper parameter만 결정하면 되는 장점이 있습니다. 그림과 같이 밀도가 높은 곳일수록 score값이 작게 되며 밀도가 높은 곳은 score값이 커지며 이상치일 확률이 높아집니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfNjEg/MDAxNjY4MzkzNzk5Mjc1.3_6kEKPMB0cgBIEnp-_TgO-i6rDE-vxCl6rYF9arrgMg.1vv7DVYylPZMCbqfnFA-R40BcBtQDSr0TlvzxtrgqRsg.PNG.dhyoo9701/14.PNG?type=w773"></p>

1. k-distance of an object p  
A 데이터로부터 k번째로 가장 가까운 데이터까지의 거리를 뜻합니다. 

2. k-distance neighborhood of an object p  
$N_k(p)$는 k-distance안에 들어오는 object의 집합을 나타내며 이는 k-distance보다 작거나 같은 거리를 가집니다. 

3. reachability distance  
A와 B까지 거리와 k-distance중 큰 값을 사용하는 것입니다.   
$reachability - distance_k(A,B) = max{k -distance(B), dist(A,B)}$

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTMz/MDAxNjY4MzkzNzk5MzI1.JCKL_zNeghrcxSUefYGG6dYhtspjWUMYFBTLvuxW8YIg.X8cJigonhAefrgAHjQTJy1tVxhMXA8VIFfgLaOhX_zUg.PNG.dhyoo9701/15.png?type=w773"></p>

4. Local reachability density of an object p  

$$lrd_k(p)=[{\frac{N_k(p)}{\sum_{o\in N_k(p)}(reah\_dist(p,o))}}]$$  

분자는 k-distance안의 개체 수이고, 분모는 p에서 다른 오브젝트까지의 reachability distance입니다. 즉, 주변의 dense를 고려한 점 p에서의 neighbor들과의 적당한 거리를 나타낼 수 있습니다.    

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjgz/MDAxNjY4MzkzNzk5MzM5.4JT2hqEm7_2N-bK7AzJqr_Vv94Zqd0rnfHPijCUG4icg.un1ex7wPS9dHyFiUL_gpCcy9NYNkR1JAD3JtpcUSlYAg.PNG.dhyoo9701/16.png?type=w773"></p>

case1의 경우, 파란 점의 lrd는 초록점들과의 거리의 평균, 초록점들의 k_distance들의 평균의 역수가 됩니다. 그러나 평균거리가 작을 것이기 때문에 lrd값은 커집니다.  
case2의 경우, 평균거리는 상당히 크기 때문에 lrd는 작은 값을 갖게 됩니다. 

1. Local outlier factor of an object p  

$$LOF_k(p)=\frac{\sum_{o\in N_k(p)}\frac{lrd(o)}{lrd(p)}}{N_k(p)}$$

주변의 점들 o와의 dense(lrd)를 비교하여 평균을 낸 것이다. p의 평균거리/o의 평균거리를 구한고 이를 평균낸 것입니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTQg/MDAxNjY4MzkzNzk5Mzk5.Rhxjpod60LYHt_kWuJV-vtvd_uEI8G2i46y2w7z1U6og.pxkmQF_BiLzabt3KVHWONd9IVpXHExcmldtpqjTV4ucg.PNG.dhyoo9701/17.png?type=w773"></p>

case1이나 case3 같이 평균 거리가 크지 않은 경우 lof는 1에 근사한 값이 나오게 됩니다. 그러나 case2와 같이 주변 점들이 가진 평균 거리에 비해 평균 거리가 더 긴 파란점의 경우 lof가 1보다 더 크게 나오기 쉽습니다. 즉, lof가 1에 근사하면 정상데이터, 1보다 크면 이상치로 판단할 수 있습니다. 

------------

# Distance-based Anomaly Detection
## 1. k-Nearest Neighbor-based Approach
이 기법은 잘 알려진 인스턴스 기반(instance based) 학습 분류 알고리즘인 k-최근접이웃(k-Nearest Neighbor, KNN)을 응용한 기법입니다. 

- 특정 인스턴스의 Novelty score는 k개의 최근접 이웃 인스턴스들의 거리정보(다양한 거리정보 방법 사용)에 기초하여 계산됩니다. 
- 일반적인 범주에 대한 사전확률분포를 가정하지 않아 인접한 데이터의 범주를 기준으로 분류하게 됩니다. 

Novelty score를 위한 거리정보는 다음과 같습니다. 

- Maximum distance to the k-th nearest neighbor  
가장 먼거리를 novelty score로 표현합니다.

<center>
$$d_{max}^k = k(x) = ||x - z_k(x)||$$
</center>

- Average distance to the k-nearest neighbors
전체 거리의 평균을 novelty score로 표현합니다.

<center>
$$d_{avg}^k = \gamma(x) = \frac{1}{k}\displaystyle\sum_{j=1}^{k}{||x-z_j(x)||}$$
</center>

- Distance to the mean of the k-nearest neighbors
점들의 mean vector와 인스턴스와의 거리를 novelty distance로 표현합니다. 이웃들이 어디에 위치해있는지 고려할 수 있는 장점이 있습니다. 

<center>

$$d_{mean}^k = \delta(x) = ||x-\frac{1}{k}\displaystyle\sum_{j=1}^{k}{z_j(x)||}$$

</center>

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTI3/MDAxNjY4MzkzNzk5NDI0.bHFVjd03rc9uiXaq7bv3NKrKr6Kf3sSrovZXnmpnknUg.dNgRcyXYCCfHZ7oopGY9yPPcu9Q00whJpHCUe2WfX_Eg.PNG.dhyoo9701/18.png?type=w773" height=340></p>
  
<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjUg/MDAxNjY4MzkzNzk5NTA2.KvU54pZVybuLL2ZUTu7rlCY6nGGFWy2oTWo2usS827Yg.XswbMnH0pFyJCTIvdmLX_s-iMKszhP5IqKJenbFd6igg.PNG.dhyoo9701/19.png?type=w773" height=350></p>

이 그림은 앞선 수식을 기반으로 novelty score를 구한 것입니다. 빨간색 동그라미가 일반적인 novelty이며 삼각형이 일반 데이터입니다. A에서는 average distance에서만 제대로 novelty detection하였으며, B에서는 어떠한 방법도 제대로 novelty detection하지 못했습니다. 

이러한 문제점을 해결하기 위해 기존 계산 방법에 convex hull을 고려하였습니다. convex hull이란 주어진 점이나 영역을 포함하는 가장 작은 블록 집합입니다. 이웃들끼리 연결했을 때 그 안에 있으면 거리가 0, 밖이면 거리가 0이상이 되게 되는 것입니다. 즉, 이웃들과의 convex combination과의 거리를 계산하겠다는 뜻입니다. 

- Average distance to the k-nearest neighbors
<center>

$$d_{avg}^k = \gamma(x) = \frac{1}{k}\displaystyle\sum_{j=1}^{k}{||x-z_j(x)||}$$

</center>

- Convex distance to the k-nearest neighbors
<center>

$$d_{c-hull}^k = ||x-\displaystyle\sum_{j=1}^{k}{w_iz_i(x)||}$$

</center>

- Put the penalty term using the convex distance for those instances located outside the convex hull of its k-nearest neighbors

<center>

$$d_{hybrid}^k = d_{avg}^k  \times \frac{2}{1+exp(-d_{c-hull}^k)}$$

</center>

첫번째 식은 전체 거리의 평균을 novelty score로 나타내는 앞선 식과 동일합니다.  
두번째 식은 이웃들 내에 위치하면 novelty score를 0, 밖에 위치하면 패널티를 부여하는 방식입니다.  
마지막 식은 첫번째와 두번째 식을 더한 모양입니다. convex 밖에 있다면 최대 2배의 패널티를 주겠다는 말과 같습니다. hybrid novelty score의 장점은 계산량은 앞선 distance based 방법론과 비슷하지만 성능이 좋다는 장점이 있습니다.  
첫번째와 두번째 방법은 아래 그림과 같습니다.

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfNTMg/MDAxNjY4MzkzNzk5NDY0.OQ8k-rNn4ZYoYdhGZzPvI4JukZLB5-G96f4xa8DUWiQg.z-ePkGhRew5fUTPa1EIISeVgeBCWaXde7hU_NWJPs1Mg.PNG.dhyoo9701/20.png?type=w773" height=350></p>

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjE4/MDAxNjY4MzkzNzk5NTM3.QR9lbLvAUdBhCfPt3FZngi5e4pLV5CNsixfBk5mPeDQg.G2Vd-TwGpAxHNC3OfwL3RGb-fxYJzS-th_R0W91N8Iwg.PNG.dhyoo9701/21.png?type=w773" height=340></p>

위 그림을 보면 hybrid novelty score에서는 모두 적합하게 hybrid novelty detection하는 것을 알 수 있습니다.

## 2. K-means clustering based novelty detecion
이 기법은 주어진 데이터를 k개의 군집으로 묶는 군집화 알고리즘인 k-means clustering을 응용한 것입니다. k-means clustering은 k개의 군집수를 정하고 군집의 중심이 될 k개의 점을 데이터중에서 임의로 선택합니다. 일반적인 유클리드 거리를 이용하여 모든 인스턴스들을 각각 자신들에게 가장 가까운 군집에 할당하고 각 군집에 속한 인스턴스들의 평균을 계산합니다. 다음으로 구한 평균값을 군집의 새로운 평균값으로 사용하고 평균값이 더이상 변화가 없을 때까지 반복합니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTQg/MDAxNjY4MzkzNzk5NTQ4.q3AuwoAJ2v26TvqWvnb4ji09OHN7qq85lLkj-N4r3jkg.LGUu6pgggh92mEu4hORNMojmsnq-pPbVQQr6YhbpP_Ug.PNG.dhyoo9701/22.png?type=w773" height=500></p>

k값은 임의로 지정하는 hyperparameter이지만 실루엣 너비, GAP 통계량과 같은 최적군집수 결정 방법을 사용하여 정하기도 합니다. 


## 3. PCA-based novelty detection
이 기법은 reconstruction기반의 novelty detection입니다. 원래의 2차원 데이터의 분산을 반영하여 1차원으로 줄인 다음, 다시 2차원으로 reconstruct하여, 본래의 2차원 데이터와 비교하여 novelty score로 계산합니다. 원래의 데이터와 멀수록 novelty score를 크게하고 가가이 있을 수록 novelty score를 작게 합니다.  
아래 그림에서는 1번 점은 novelty score가 높고, 2번은 novelty score가 낮다고 할 수 있습니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTAw/MDAxNjY4MzkzNzk5NjEz.rbVIhM1xShuPtn5sfwQFMvEcBAjTCGC0YNEOi3VZzOEg.2Q7_y4Ig8B8BdCzVSqN3VHty7V83lSV8_0TFRwYuf3Mg.PNG.dhyoo9701/23.PNG?type=w773" height=300></p>


# Model-based Anomaly Detection

## Auto-Encoder for Novelty Detection
Auto-Encoder는 neural network 모델의 한 종류로, input data를 받았을 때 input data를 잘 표현하는 latent vector를 만드는 것이 목적입니다. encoder를 통해 입력 데이터에 대한 특징을 추출해내고, 이 결과를 가지고 neural network를 역으로 붙여 원본 데이터를 생성해내게 됩니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjI1/MDAxNjY4MzkzNzk5NjQy.PNU2c30afbSfTrE8t8pSseE-R5l6H0ZUu2iVs6wKa3Qg.z5a_RLeuN33mubyOZNjBEyR8nVBvOnOv5Nymsn6TqyEg.PNG.dhyoo9701/24.png?type=w773"></p>
  
<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTk2/MDAxNjY4MzkzNzk5Njc4.LEUfseBksvrbRfKjYeMfZsIubyS4RnExar9S6ppa7Vcg.mqn-JOOssbehS9LYlCNTKL_PyxooggleJMHC_lLlCDIg.PNG.dhyoo9701/25.png?type=w773"></p>

Auto-Encoder는 그림과 같이 output layer를 통해 나온 모델의 예측된 데이터와 실제 데이터의 차이를 loss function으로 정하고, 이 loss를 줄이는 방향으로 모델을 훈련시킵니다. 

Novelty detection에서는 Auto Encoder를 reconstruction error가 데이터의 이상치를 잡아내는데 사용합니다. Auto-Encoder는 훈련 과정에서 훈련 데이터와 똑같은 형태의 데이터를 예측하기 위해 훈련 데이터의 일반적인 특징을 배우게 됩니다. 정상 데이터를 input으로 받았을 경우 reconstruction error가 낮지만, 이상치 데이터를 input으로 받을 경우 reconstruction error가 높게 됩니다. 즉, reconstruction error가 높으면 outlier, 낮으면 정상 데이터로 분류하는 방식으로 Auto-Encoder가 사용됩니다. 


## One-Class SVM&SVDD

### One-Class SVM

One-class SVM은 데이터를 feature 공간에 매핑하고 원점과의 거리를 최대화 할 수 있는 hyperplane을 찾는게 목적입니다. hyperplane 아래에 위치하면서 원점과 가까운 데이터는 outlier, hyperplane 위에 있으면 정상 데이터로 분류하게 됩니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjMy/MDAxNjY4MzkzNzk5NzEx.tpIv5y5CvgqHGkQobFEeCallj5NHrxAzkFKiNqfbXl8g.8shWHrGrt2XOnv8FIHjpWL1gXk1jZJCVhnNdfQydecsg.PNG.dhyoo9701/26.png?type=w773"></p>

One-class SVM의 수식은 다음과 같습니다. 기존의 SVM 목적식에서 마진을 최대화하는 부분이 비슷합니다. 하지만 one-class SVM에서는 단순히 마진을 최대화할 경우 decision boundary가 원점에서부터 음 혹은 양의 방향으로 무한히 발산하게 됩니다. 이를 해결하기 위해 one-class SVM에서는 $-\rho$를 통해서 decision boundary가 원점으로부터 양의 방향으로 최대한 멀어지도록 제약을 더해주게 됩니다. 또한, decision boundary 아래에 존재하는 객체들에 대해 패널티를 가하고 이를 최소화하도록 제약을 추가합니다.  
해당 목적식의 제약식을 Lagrangian problem으로 변환하고 KKT조건으로 문제를 풀게 됩니다. 해당 목적식으로부터 w를 얻고 decision function을 얻을 수 있게 됩니다. 새로운 데이터가 입력되었을 떄 그 값이 +이면 정상, -이면 outlier 데이터로 분류하게 됩니다. 

<center>

$$min_w  \frac{1}{2}\ \lVert w\rVert ^2+ \frac{1}{\nu l}\sum_{i=1}^l\xi_i-\rho\\
s.t. \quad \mathbf{W}\cdot\boldsymbol{\Phi}(\mathbf{X_i})\ge\rho-\xi_i\\
i=1,2,...,l, \qquad \xi\ge0\\$$

</center>

- Decision function

<center>

$$f(\mathbf{X_i})= sign(\mathbf{W}\cdot\mathbf{\Phi(\mathbf{X_i})-\rho})$$


</center>

- Lagrangian problem

<center>

$$L=\frac{1}{2}\ \lVert w\rVert ^2+ \frac{1}{\nu l}\sum_{i=1}^l\xi_i-\rho-\sum_{i=1}^l\alpha_i(\mathbf{W}\cdot\boldsymbol{\Phi}(\mathbf{X_i})-\rho+\xi_i)-\sum_{i=1}^l\beta_i\xi_i$$

</center>

$l, X_i$는 데이터가 주어지면 알 수 있는 값입니다. 우리가 식으로 부터 hyperplane을 도출하고자 할 떄 구해야하는 미지수는 $W, \xi, \rho$입니다. 따라서 목적식을 3개의 미지수에 대해서 편미분하여 KKT 조건을 유도하게 됩니다. 

<center>

$$\frac{\partial L}{\partial \mathbf{w}}=\mathbf{w}-\sum_{i=1}^l\alpha_i\mathbf{\Phi(\mathbf{X_i})}=0\quad \Rightarrow\quad \mathbf{w}=\sum_{i=1}^l\alpha_i\mathbf{\Phi(\mathbf{X_i})}$$

$$\frac{\partial L}{\partial \xi_i}=\frac{1}{\nu l}-\alpha_i-\beta_i=0\quad \Rightarrow\quad \alpha_i=\frac{1}{\nu l}-\beta_i   \\
\frac{\partial L}{\partial \rho}=-1+\sum_{i=1}^l\alpha_i=0\quad \Rightarrow \quad \sum_{i=1}^l\alpha_i=1$$

</center>

primal의 목적식을 dual 형태로 바꾸게 됩니다. 이 dual을 최대화하는 최적화 문제로 풀게 되면 primal의 최소값을 구할 수 있습니다. 

<center>

$$min \; L= \frac{1}{2}\sum_{i=1}^l\sum_{j=1}^l\alpha_i\alpha_j\mathbf{\Phi(\mathbf{X_i})}\cdot\mathbf{\Phi(\mathbf{X_j})} \\
s.t. \quad \sum_{i=1}^l\alpha_i =1 , \quad 0\le \alpha_i\le\frac{1}{\nu l }$$

</center>

최적해를 얻는 과정에서 complementary slackness 조건에 따라 lagrangian multiplier의 케이스가 나뉘게 됩니다. 

- Case 1: $\alpha_i=0 \Longrightarrow$ a non support vector
- Case 2: $\alpha_i = \frac{1}{vl} \longrightarrow \beta_i=0 \longrightarrow \xi_i>0 \Longrightarrow$ Support vector (outsider the hyperplane)
- Case 3: $0<\alpha_i<\frac{1}{vl} \longrightarrow \beta_i>0 \longrightarrow \xi_i=0 \Longrightarrow$ Support vector (on the hyperplane)   

Case 1은 $\alpha_i$=0인 경우는 $W\cdot\Phi(X_i)-\rho+\xi\ne0$을 의미하기 때문에 데이터 포인트는 support vector가 아닙니다. 

Case 2는 $\alpha_i=\frac{1}{vl}$을 만족하는 경우, $W\cdot\Phi(X_i)-\rho+\xi=0$을 의미하기 때문에 support vector이며 $\beta_i=0$을 만족하게 되어 $\xi_i>0$을 만족합니다. 이는 이 데이터 포인트가 hyperplane 밖의 support vector임을 의미합니다. 

Case 3는 Case 2와 마찬가지로 데이터포인트가 support vector이며, $\beta>0$을 만족하며, 이는 $\xi_i$=0으로 hyperplane임을 의미합니다. 


$\upsilon$의 역할은 다음과 같습니다. 
1. hyperplane 밖에 있을 수 있는 최대한의 support vector의 개수를 정해주는 비율은 $\alpha_i \leq\frac{1}{vl}$이므로 $\alpha_i$의 최대값은 $\frac{1}{vl}$입니다.
2. 데이터가 가져야하는 최소한의 support vector의 개수를 정해주는 비율입니다. 첫번째 역할의 연장선에서 설명할 수 있는데, $\alpha_i$들이 모두 $\frac{1}{vl}$인 경우가 support vector의 전체 개수는 가장 적은 경우라고 할 수 있습니다. 하나라도 $\alpha_i<\frac{1}{vl}$인 경우가 존재한다면 $\sum_{i=l}^l\alpha_i=1$을 만족하기 위해 그 차이를 메우기 위한 0이 아닌 $\alpha_i$들이 존재해야 하기 때문입니다. 
3. 최소 support vector가 $vl$개 있어야 하는 것을 의미합니다. 따라서 정상 영역의 범위를 넓게 만들고 싶다면 $\upsilon$을 작게 잡아야합니다. 아래 예시 그림을 보면 왼쪽의 경우 정상영역의 범위가 오른쪽보다 넓게 형성되어 있는데 왼쪽의 경우가 오른쪽보다 $\upsilon$을 더 작게 잡았다고 할 수 있습니다.

One-class SVM도 SVM과 마찬가지로 kernel함수를 사용할 수 있습니다. kernel trick을 적용해 고차원에 매핍해 hyperplane을 찾을 수 있습니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTY0/MDAxNjY4MzkzNzk5NzQw.iK0gwWbVPagtzJT4fP1kjI3s0f-002r48vtLI8n9qZQg.qZQILZkxen1-_G7mM1qMam6chTR5JYTfucJoOvCYhrcg.PNG.dhyoo9701/27.png?type=w773"></p>


### SVDD(Support Vector Data Description)
SVDD는 정상 데이터를 둘러싸는 가장 작은 초구를 찾는 것이 목적입니다. 범주를 나눌 수 있는 hyperplane을 찾는 one-class SVM과 달리 SVDD는 초구를 찾게 됩니다.  

SVM의 마진 대신 원의 반지름을 최소화하는 구조입니다. 다만 원의 크기를 최소화할 경우 decision boundary가 작은 한점으로 무한히 수렴하게 됩니다. 따라서 클래스로 인정받지 못하는 샘플을 최소화하도록 제약을 추가하게 됩니다. SVDD에서는 벗어난 객체와 초구와의 거리를 정확하게 계산 할 수 없기 때문에, 그냥 벗어난 것에 대해 패널티를 주는 개념입니다.   
SVDD의 수식은 다음과 같습니다. 

- Primal problem

<center>

$$min_{R,\mathbf{a},\xi_i} R^2 + C\sum_{i=1}^l\xi_i$$

$$s.t. \lVert \mathbf{\Phi(\mathbf{X_i})-\mathbf{a}} \rVert^2 \le R^2+\xi_i, \quad \xi_i\ge0,\; \forall i$$

</center>

- Desicion function

<center>
$$f(\mathbf{X})=sign(R^2-\lVert \mathbf{\Phi(\mathbf{X_i})-\mathbf{a}} \rVert^2)$$
</center>

- Primal Lagrangian problem

<center>
$$L=R^2 + C\sum_{i=1}^l\xi_i- \sum_{i=1}^l\alpha_i(R^2+\xi_i-(\mathbf{\Phi(\mathbf{X_i})\mathbf{\Phi(\mathbf{X_i})-2\cdot a\cdot \mathbf{\Phi(\mathbf{X_i})+a\cdot a}}}))-\sum_{i=1}^l\beta_i\xi_i(7)$$

$$\alpha_i \ge0, \; \beta_i \ge0$$


</center>

- KKT condition

<center>

$$\frac{\partial L}{\partial R}=2R-2R\sum_{i=1}^l\alpha_i=0\quad\Rightarrow\quad\sum_{i=1}^l\alpha_i=1$$


$$\frac{\partial L}{\partial \mathbf{a}}=2\sum_{i=1}^l\alpha_i\cdot\mathbf{\Phi(\mathbf{X_i})-2\mathbf{a}\cdot\sum_{i=1}^l\alpha_i=0}\quad \Rightarrow\quad \mathbf{a}=\sum_{i=1}^l\alpha_i\cdot\mathbf{\Phi(\mathbf{X_i})}$$

$$\frac{\partial L}{\partial \xi_i}=C-\alpha_i-\beta_i=0 \quad \forall i$$


</center>

- Dual Lagrangian problem

<center>

$$L=R^2 + C\sum_{i=1}^l\xi_i- \sum_{i=1}^l\alpha_i(R^2+\xi_i-(\mathbf{\Phi(\mathbf{X_i})\mathbf{\Phi(\mathbf{X_i})-2\cdot a\cdot \mathbf{\Phi(\mathbf{X_i})+a\cdot a}}}))-\sum_{i=1}^l\beta_i\xi_i$$

$$=R^2-R^2\sum_{i=1}^l\alpha_i+\sum_{i=1}^l\xi_i(C-\alpha_i-\beta_i)$$

$$+\sum_{i=1}^l\alpha_i\mathbf{\Phi(\mathbf{X_i})}\cdot\mathbf{\Phi(\mathbf{X_i})}-2\sum_{i=1}^l\sum_{i=1}^l\alpha_i\alpha_j\mathbf{\Phi(\mathbf{X_i})}\cdot\mathbf{\Phi(\mathbf{X_j})}\\+\sum_{i=1}^l\sum_{i=1}^l\alpha_i\alpha_j\mathbf{\Phi(\mathbf{X_i})}\cdot\mathbf{\Phi(\mathbf{X_j})}$$

$$max\;L=\sum_{i=1}^l\alpha_i\mathbf{\Phi(\mathbf{X_i})}\cdot\mathbf{\Phi(\mathbf{X_i})}-\sum_{i=1}^l\sum_{i=1}^l\alpha_i\alpha_j\mathbf{\Phi(\mathbf{X_i})}\cdot\mathbf{\Phi(\mathbf{X_j})}\qquad (0\le\alpha_i\le C)$$

</center>

이를 Minimization 문제로 바꾸기 위해서 단순히 -1을 곱하면 됩니다. 최종적으로 dual problem은 다음과 같습니다.

<center>

$$min\;L =\sum_{i=1}^l\sum_{i=1}^l\alpha_i\alpha_j\mathbf{\Phi(\mathbf{X_i})}\cdot\mathbf{\Phi(\mathbf{X_j})}-\sum_{i=1}^l\alpha_i\mathbf{\Phi(\mathbf{X_i})}\cdot\mathbf{\Phi(\mathbf{X_i})} \qquad (0\le\alpha_i\le C)$$
</center>

따라서 $||w||=1$인 경우, SVDD와 one-class SVM은 같은 알고리즘이 되게 됩니다.

## Isolation Forest

Isolation Forest는 하나의 이상치를 고립시키는 Tree를 생성하는 것이 목적입니다. 이상치는 개체수가 적으며, 정상 데이터와는 특정 속성 값이 많이 다를 가능성이 높습니다. 아래 그림과 같이 사용되는 선의 개수가 적을수록 outlier에 가까워집니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTYx/MDAxNjY4MzkzNzk5Nzcy.4ah1rxmgCQyGOR-rdaOHCKV9lHZUvtI7-Onvreya0jEg.Shw1p65N_K9ir2zQxn7VxJbIYr_p3_NHo2jR_jhDoLQg.PNG.dhyoo9701/28.png?type=w773"></p>

아래 그림과 같이 각각의 데이터를 다른 데이터들로부터 분리시키는 과정은 하나의 Isolation tree로 표현할 수 있습니다. 그리고 이 분리 과정을 랜덤하게 여러번 반복하면 isolation forest가 됩니다. 이 때 여러번의 랜덤한 분리과정에서 꾸준히 적은 선으로 분리되는 데이터는 outlier로 분류되고, 꾸준히 많은 선으로 분리되는 데이터는 정상데이터로 분류됩니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTMz/MDAxNjY4MzkzNzk5ODAy.kKNnJX_R1YZ9UYV1FNqhtsObEoTuH81dAnTp4oK3qIUg.f1uOpmQKw6wtlwx2e2XyG_sKFTSruZGcCFLkEk5n2aAg.PNG.dhyoo9701/29.png?type=w773"></p>

즉, Tree에서 path length가 짧을수록 novelty score는 1에 가까워지고, path length가 길수록 novelty score는 0에 가까워집니다. 특히 0~1사이의 값을 갖기 때문에 데이터셋이 다르더라도 novelty score가 가지는 의미를 이해하기 쉽습니다. 데이터셋이 다르더라도 상대적인 해석이 어느정도 가능하다는 장점이 있습니다.

- Path Length

<center>

$$c(n) = 2H(n-1)-\frac{2(n-1)}{n}, H(i)=ln(i) + 0.5772156649$$ (Euler's constant)

</center>

- Novelty Score

<center>

$$s(x,n) = 2^-\frac{E(h(x))}{c(n)}$$

</center>

Novelty Score에서 E(h(x))는 Isolation forest에서의 데이터 x의 평균 거리를 나타냅니다. 즉, 선이 많이 필요한 정상데이터는 E(h(x))값이 높으며, 전체 novelty score는 낮게됩니다. 반대로 선이 적게 필요한 outlier의 경우 E(h(x))값이 낮으며, 전체 novelty score는 높아집니다.

Isolation forest는 직관적으로 2차원 공간에서 생각했을 때 단순히 선을 그어 이상치를 고립시키면 이상치가 아닌 객체의 영역에 대해서도 분기를 많이 해야합니다. 아래 그림과 같이 해당 집단들 안에 이상치가 있다고 가정하면 정상 범주에 대해서도 분기가 많이 수행됩니다. 특히, 오른쪽 그림과 같은 경우는 수직 선으로 분기해서 영역을 분기하기엔 한계가 있습니다.

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMjI4/MDAxNjY4MzkzNzk5ODUz.rPQK46PVD9pEK73JH9wXOTjZaPdPF9G0oWSvGTq-Uxwg.U0ri35JPkpUhlc0LKFikhLm5FPkXPUwXCHVcrg55d0sg.PNG.dhyoo9701/30.png?type=w773"></p>


그래서 사용하는 방법이 Extended Isolation forest입니다. 앞선 문제점이었던 데이터에서 이상치가 위치하는 부분을 고립시키기 위해 정상범주에 분기를 많이해야한다는 단점을 해결하기 위한 방법입니다. 데이터를 분기할때 수직으로 split하는게 아닌 기울기를 가질 수 있도록해서 다양하게 고립시키는 방법입니다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjExMTRfMTky/MDAxNjY4MzkzNzk4ODEz.5PKfkxM8gGy24ET6VF0Ch5WzsS3_LTcjWv84-Nogpgwg.O6S455zISPzbn8f6Qn0SIj37rpUkoPyZw01-AwBMbB4g.PNG.dhyoo9701/32.png?type=w773"></p>

위 그림과 같이 기울기가 있는 선으로 분기하였더니 정상 범주 영역에 선의 밀도가 가시적으로 개선된 것을 확인할 수 있습니다. 
