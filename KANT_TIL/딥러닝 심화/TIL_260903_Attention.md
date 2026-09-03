# 용어 정의

`attention`: 현재 토큰이 다른 토큰들을 살펴보고, 필요한 정보를 서로 다른 비율로 가져오는 방식
`Query`: 지금 무엇을 알고 싶은가?
`Key`: 이 후보는 질문과 어떤 관련이 있는가?
`Value`: 이 후보에서 실제로 가져올 정보는 무엇인가?

## 1. 생성형 Transformer 학습 방식

- 생성형 Transformer의 사전학습은 보통 자기지도학습(Self-Supervised Learning)이다.
- 사람이 직접 label을 붙이지 않지만, 입력 데이터 자체에서 다음 token 등의 target을 만들어 학습한다.
- Instruction Tuning은 질문-답변 형태의 지도학습으로 볼 수 있다.
- 분류 문제인지 여부와 지도학습 여부는 별개이며, 핵심은 학습 target의 존재 방식이다.

## 2. Scaled Dot-Product Attention

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- `Q`: 무엇을 찾을지 표현
- `K`: 각 token이 어떤 정보를 가지고 있는지 비교하기 위한 표현
- `V`: 실제로 가져와 섞을 정보
- Attention Weight: 각 token을 얼마나 참고할지 나타내는 비율
- `V`는 입력 `X`에 학습 가능한 `W_V`를 곱해 만든다.

$$V=XW_V$$

- `W_V`는 학습되는 모델 파라미터이다.
- Attention Weight는 입력에 따라 매번 계산되는 값이다.

## 3. Attention Weight와 Value

- Attention Weight = 누구를 얼마나 참고할지
- Value = 그 token에서 실제로 가져올 정보

$$\mathrm{Output}=\sum_i a_iV_i$$

예를 들어 Attention Weight가 `0.7, 0.2, 0.1`이면 각 Value를 해당 비율만큼 가중합해 Contextual Vector를 만든다.

## 4. Head

- Head = 하나의 독립적인 Attention 계산 단위
- 하나의 Head는 자기만의 `W_Q`, `W_K`, `W_V`를 사용한다.
- Single-Head Attention도 모든 Query에 대해 Attention을 계산한다.
- 첫 번째 Query 하나만 사용하는 것이 아니다.

예를 들어 token이 3개이면:

- `Q1`은 `K1`, `K2`, `K3`와 비교
- `Q2`는 `K1`, `K2`, `K3`와 비교
- `Q3`는 `K1`, `K2`, `K3`와 비교

## 5. Multi-Head Attention

- Multi-Head Attention은 Head를 여러 개 사용하는 구조이다.
- 각 Head는 서로 다른 `W_Q`, `W_K`, `W_V`를 학습한다.
- 따라서 같은 입력을 여러 projection 관점에서 바라볼 수 있다.
- 각 Head의 Attention Weight도 서로 다르게 계산된다.

핵심:

- Single-Head: `W_Q`, `W_K`, `W_V` 한 세트
- Multi-Head: `W_Q`, `W_K`, `W_V` 여러 세트

## 6. Self-Attention이 필요한 이유

Self-Attention은 각 token이 다른 token과의 관계를 직접 참고해 문맥에 맞는 의미 표현을 만들기 위해 필요하다.

- 하나의 token은 문맥에 따라 의미가 달라질 수 있다.
- 따라서 주변 token과의 관계를 함께 봐야 정확한 의미를 파악할 수 있다.

## 7. Linear Projection

`nn.Linear(D, D, bias=False)`는 입력의 마지막 차원 `D`를 같은 크기의 `D`로 선형 변환한다.

예를 들어:

- `B = 2`
- `L = 5`
- `D = 8`

입력 shape:

`[2, 5, 8]`

Linear 적용 후:

`[2, 5, 8]`

shape는 유지되고 값만 학습된 가중치로 변환된다.

`nn.Linear(8, 8, bias=False)`의 weight shape:

`[8, 8]`

weight는 처음에 랜덤 초기화되고 학습 과정에서 업데이트된다.

## 8. Linear 행렬 연산

`x.shape = [B, L, D]`이고 `weight.shape = [D, D]`이면 마지막 차원 `D`에 대해 같은 행렬 변환이 적용된다.

$$[B,L,D]\times[D,D]\rightarrow[B,L,D]$$

PyTorch의 `nn.Linear`는 내부적으로 다음 형태로 계산한다.

$$y=xW^T$$

`bias=False`이면 bias는 더하지 않는다.

## 9. Scaling에서 제곱근을 사용하는 이유

`d_k`는 Key 벡터의 차원 수이다.

$$\frac{QK^T}{\sqrt{d_k}}$$

`d_k`가 커지면 dot product 값의 크기도 커질 수 있다.

이를 `√d_k`로 나눠 크기를 조절하면 softmax가 지나치게 뾰족해지는 것을 완화하고 학습을 안정적으로 유지할 수 있다.

