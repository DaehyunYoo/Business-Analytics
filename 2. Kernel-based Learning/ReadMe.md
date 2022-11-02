# Suppor Vector Machine

## 이론

SVM은 분류, 회귀(SVR) 목적으로 사용되는 머신러닝 알고리즘이다. SVM은 분류, 회귀 및 이상치 탐지 목적을 위한 머신러닝 알고리즘중 하나이다. SVM분류기는 지정된 범주 중 하나에 새 데이터 포인트를 할당하는 모델을 구축한다. 이진 선형 분류기로 볼 수 있다. 

Support vector machine은 벡터 공간 상에 있는 데이터들을 가장 잘 나눌 수 있는 직선(hyperplane)을 찾아내는 것을 목표로 한다.  
SVM은 선형분류 목적 이외에도 SVM은 커널 트릭을 사용하여 비선형 분류를 효율적으로 수행할 수 있다. 이를 통해 입력을 고차원 형상 공간에 암시적으로 매핑할 수 있다. 선형으로 분류가 불가능한 경우엔 Kernel 함수를 도입해 분류 경계면을 탐색하게 된다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjEwMjlfMTUx/MDAxNjY3MDMxMjAzNTY3.st26a8hL1CU6mTcU3SMDzhyLcKoZwJw7AEt2CcxZQ4og.By022otz0EV6Ezbc-kjb2FA_w8YuI3tj96lLYSj-nugg.PNG.dhyoo9701/1.png?type=w773"></p>


분류 모델은 모형의 복잡도가 증가할수록 주어진 학습 데이터를 더 잘 분류할 수 있게 되지만 데이터에 있는 노이즈까지 학습하게 되면서 학습데이터만 잘 분류하는 문제가 커지게된다. 따라서 어떤 분류 모형이 실제 데이터가 주어졌을 때 이를 잘 분류하기 위해서는 학습과정에서의 오류를 낮추면서 모델의 복잡도도 줄여야 한다. 이 두가지를 모형의 구조적 위험이라 한다. 

SVM은 d차원의 데이터를 나누는 d-1차원의 hyperplane을 구하는 문제인데 이 hyperplane들에 적용하는 기준이 되는 것이 마진(margin)이다. 마진은 hyperplane에 의해 분류된 데이터포인트와 hyperplane간의 거리들 가운데 가장 짧은 거리를 의미한다.

SVM은 이론적으로 마진의 크기를 최대화하면 구조적 위험이 최소화된다. 어떤 함수의 복잡도를 측정하는 지표 즉, 얼마만큼 표현력을 가지고 있는지에 대한 지표인 VC dimension이라고 하는데 모델의 구조적 위험은 데이터로부터 얻는 위험과 모델의 복잡성으로부터 얻는 위험의 합만큼 가질 수 있다. 즉, 마진을 최대화하는 것이 결국 모델의 구조적 위험을 최소화하므로, 마진을 최대화해서 모델의 일반화 성능을 높일 수 있다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjEwMjlfMjg5/MDAxNjY3MDMxMjAzNTcx.aZK_AqYx1ppS6g3hQHK8GSVNQwIN-HM0HRPdAMA1-xsg.9xBirEOWWVBCK7fZ0qjyQpWZCYBqOLrHX-0H0U0o0RQg.PNG.dhyoo9701/2.png?type=w773"></p>


## SVM Case 1: Linear & Hard Margin
<p align="center"><img src= "https://postfiles.pstatic.net/MjAyMjEwMjlfMjI0/MDAxNjY3MDMxMjAzNTY4.XTebo2szkUuJQY2ka86y30C5_9HpsDTfoUR4BOPVoy4g.Dk8ONeDDMHX5IGQ3MZ5s7O9Q4vIdV6Ti6gO2m_62Cqog.PNG.dhyoo9701/4.png?type=w773"></p>  


<center> 

$$
min \frac {1} {2} \lVert {w} \rVert^{2}$$ 

</center> 


<center> 

$$s.t. \quad  y_i \left( w^{t}x_i +b \right) \ge 1$$ 

</center>


선형 분류를 하고자 할 때 하드마진을 사용해 SVM을 만든다면 다음과 같이 마진을 최대화하도록 목적식을 정의합니다. 해당 목적식으로부터 SVM의 최적해를 얻는 과정에서 KKT 조건에 의해 w와 b를 얻을 수 있게 됩니다. 

### Lagrange multiplier

제약이 있는 최적화 문제를 푸는 것이므로 Lagrange multiplier을 사용할 수 있다. 여기서 $\alpha$는 Lagrange multiplier이다. 

<center> 

$$
{\min{ L_{p}(w,b,{ \alpha }_{ i }) } } =\frac { 1 }{ 2 } { \left| w \right| }^{ 2 }-\sum { i=1 }^{ n }{ { \alpha }_{ i }({ y }_{ i }
({ w }^{ T }{ x }_{ i }+b)-1) }
$$ 

</center>

L을 미지수 w와 b로 각각 편미분한 값이 0이 되는 곳에서 최솟값을 갖으므로 각각을 정리하면 각각을 정리하면 w와 b를 a와 x, y에 대한 식으로 정리할 수 있다. 

Lagrange Primal problem만으로는 문제를 해결하기 어렵다. Dual Problem을 이용해 Lagrange problem을 해결할 수 있다. Dual Problem으로 생성된 수식은 다음과 같다. 

<center>

$$
\max { { L }_{ D }({ \alpha }_{ i }) } =\sum_{ i=1 }^{ n }{ { \alpha }_{ i } } -\frac { 1 }{ 2 } \sum_{ i=1 }^{ n }{ \sum_{ j=1 }^{ n }{ { \alpha }_{ i }{ { \alpha }_{ j }y }_{ i }{ y }_{ j }{ x }_{ i }^{ T }{ x }_{ j } } }
$$

$$
s.t.  \quad  \sum_{ i=1 }^{ n }{ { \alpha }_{ i }{ y }_{ i } } = 0,   { \alpha }_{ i }\ge 0,\quad i=1,...,n
$$

</center>

마지막 식이 결국 $\alpha$에 대한 maximization 문제로 정리가 되었다. 이 문제를 풀어서 $\alpha$를 구하면 $w= \sum {i=1}^{n}{ \alpha_{i}y_{i}x_{i} }$ 를 통해서 $\omega$  만족해야한다. 


### KKT Condition
KKT(Karush-Kuhn-Tucker) Condition은 다음과 같다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjEwMjlfNzAg/MDAxNjY3MDMxMjAzNTY5.wJnviFsSogclEmb8iwz-VCldC60OBzTuN8XmF9Q8N0og.vW6J9webRFdFqz0jnLOtGw0TP1dRm-sssF5XvaPIgqwg.PNG.dhyoo9701/5.png?type=w773"></p> 


## SVM Case 2: Linear & Soft Margin

Soft margin을 사용한 경우의 SVM을 살펴보면 다음과 같다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjEwMjlfMjQ1/MDAxNjY3MDMxMjAzNTY3.fQZoSZPaJGy7fsTKn8Lw8hjCnoMSURqT64LAUVFzCrQg.EvrYhUp0YJms2CoEfYAG82tDPLHXfKZO2Hlk9EItciEg.PNG.dhyoo9701/6.png?type=w773"></p>


<center> 

  
$$
\min { \frac { 1 }{ 2 } { \left\| w \right\|  }_{ 2 }^{ 2 } } +C\sum _{ i=1 }^{ n }{ { \xi  }_{ i } }
$$ 

$$
s.t. \quad { y }_{ i }({ w }^{ T }{ x }_{ i }+b)\ge 1-{ \xi  }_{ i },\quad { \xi  }_{ i }\ge 0
$$ 

  
</center>

목적식을 보면 Hard margin일때와의 차이는 조건을 만족시키지 못하는 경우 $\xi$를 통해 패널티를 부여한다. $\xi$는 + plane 혹은 - plane으로부터 해당 범주와의 거리이다. 즉, Hadr margin일 때보다 찾아야 하는 미지수가 하나 더 늘어난 것이다. 
해당 목적식으로부터 SVM의 최적해를 얻고자 하면, 결국 Dual Problem으로 넘어와 제약식의 범위만 바뀌게 된다. 
이어 KKT condition에 의해 다음과 같은 식을 얻을 수 있다.  

<center>

$$
\alpha \left( y_i \left(w^{T}x_i + b \right)-1+\xi_{i} \right) = 0
$$

</center>

위 식에 따라 support vector는 $\alpha \ne 0$ 일때 임을 알 수 있으며 Soft margin일 경우 3가지 케이스로 나눌 수 있다. 

<center>

$$
C- \alpha_{i}-u_{i} = 0 \; and \; u_{i}\xi_i = 0
$$

</center>

<center>

1. $\alpha_i=0$ 이면 관측치는 non-support vector  
  
2. $$\0<alpha_i<C$$ 이면 관측치는 마진 위의 support vector
  
3. $$\alpha_i=C$$ 이면 관측치는 마진 밖의 vector

</center>
  
### C에 따른 영향
C는 Regularization을 해준다. Large C이면 $\xi$가 작아지고 마진이 좁아져 적은 수의 support vector가 생성된다. 반대로 Small C이면 $\xi$가 커지고 마진이 커져 많은 수의 support vector가 생성된다.  

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjEwMjlfMjE5/MDAxNjY3MDMxMjAzNTcz.G9xLodwjlIRzBzvsRTXid-tnJ_BSeJZgHgJIDbluGrEg.oA9FQCShME_ENdVGppjYnxPE2kvVVWetBDgpeP-3sfsg.PNG.dhyoo9701/7.png?type=w773"></p>

## SVM Case 3: non Linear & Soft Margin

### Kernel SVM 
본래의 차원에서 선형 분리가 불가능할 경우, 더 높은 고차원에서 맵핑하였을 때 가능한 경우가 있다. 이러한 아이디어를 SVM에 도입하여 입력데이터를 고차원 공간으로 보낸 뒤에 결정 경계면을 찾고자 하는게 Kernel SVM이다. 

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjEwMjlfMjMz/MDAxNjY3MDMxMjAzNzQx.tm3J5ZY6SYPzyS03LiheVwmZDC5UkluwgjaNRPE7u2sg.Xf1mkgAWNKxmO4DCODlIZtcqXBNkRpCUEk5Eo9XdH7wg.PNG.dhyoo9701/8.png?type=w773"></p>

### Nonlinear Soft Margin

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjEwMjlfMjU2/MDAxNjY3MDMxMjAzNzQ1.oIQ3EBZdP46u9M3e8tVqvMSHorK0vPY9NFhnc2rHRqsg.NWoJEQjlJX1fU7M7THdowkMt5OigbeVeFP7IbrXfReMg.PNG.dhyoo9701/9.png?type=w773"></p>

<center>

$$
\min \frac{1}{2} \parallel w \parallel + C \sum_{i=1}^N \xi_i
$$

$$
s.t. \quad y_i \left( w^T \Phi \left(x_i \right) + b   \right) \ge 1-\xi_i, \;\; \xi_i \ge 0, \forall i
$$

</center>

앞선 선형의 두 가지 경우와 목적식은 거의 동일하다. 차이점은 제약식에 $x_i$를 감싸는 $\phi(x_i)$가 생겼다는 점이다. 이 $\phi$가 저차원의 데이터를 고차원으로 변형해주는 mapping function인 것이다. 

입력데이터를 고차원 공간에 맵핑하면 모든 관측치에 대해 고차원으로 맵핑하고 내적하기 때문에 연산량이 폭증하는 문제가 발생한다. 이러한 문제를 해결하는 것이 바로 Kernel Trick이다.

### Kernel Trick

Kernel Trick은 고차원 맵핑과 내적을 한번에 해결하고자 하는데 목적이 있다. Kernel 함수는 Kernel이라는 고차원 공간에서의 내적 특징을 충분히 보존할 수 있는 함수이다. 
Kernel 함수로 주로 사용되는 것은 다음과 같다. 

<center>

<p align="center"><img src="https://postfiles.pstatic.net/MjAyMjEwMjlfMjEg/MDAxNjY3MDMxMjAzNzQ5.G1z94shYpF4REJa-_djzBd2Rc76CoT4fTSXg-FpmUkYg.ZxSZcokFltB1G7LZU6jI7hV6MISE7-C_Vu7icOLi538g.PNG.dhyoo9701/10.png?type=w773"></p>

</center>


