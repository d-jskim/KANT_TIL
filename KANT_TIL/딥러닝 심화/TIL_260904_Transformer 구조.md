## 1. 생성형 Transformer 학습 방식

- 생성형 Transformer의 사전학습은 주로 자기지도학습(Self-Supervised Learning)이다.
- 사람이 직접 label을 붙이지 않지만 데이터 자체에서 다음 token 등의 target을 만들어 학습한다.
- Instruction Tuning은 질문-답변 형태의 지도학습으로 볼 수 있다.
- 분류 문제인지 여부와 지도학습 여부는 별개이다.

## 2. Self-Attention

Self-Attention은 각 token이 다른 token과의 관계를 참고하여 문맥이 반영된 표현을 만드는 과정이다.

- `Q`: 무엇을 찾을지 나타내는 표현
- `K`: 어떤 정보를 가지고 있는지 비교하기 위한 표현
- `V`: 실제로 가져올 정보
- Attention Weight: 각 token을 얼마나 참고할지 나타내는 비율

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V$$

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- `W_Q`, `W_K`, `W_V`는 학습되는 모델 파라미터이다.
- Attention Weight는 `Q`, `K`를 이용해 입력마다 동적으로 계산된다.
- `√d_k`로 나누는 이유는 dot product 값이 너무 커져 softmax가 지나치게 뾰족해지는 것을 완화하기 위해서이다.

핵심:

**Attention Weight는 누구를 얼마나 참고할지 정하고, Value는 실제로 가져올 정보를 제공한다.**

## 3. Single-Head와 Multi-Head Attention

### Single-Head

Single-Head도 모든 Query에 대해 Attention을 계산한다.

token이 3개라면:

```text
Q1 → K1, K2, K3
Q2 → K1, K2, K3
Q3 → K1, K2, K3
```

즉 첫 번째 Query 하나만 사용하는 것이 아니다.

### Multi-Head

- Head = 하나의 독립적인 Attention 계산 단위
- Single-Head: `W_Q`, `W_K`, `W_V` 한 세트
- Multi-Head: 여러 세트의 `W_Q`, `W_K`, `W_V`
- 각 Head가 서로 다른 projection을 학습하므로 같은 입력을 여러 관점에서 볼 수 있다.
- Head마다 Attention Weight도 다르게 계산된다.

## 4. Linear Projection과 Shape

예:

```python
B, L, D = 2, 5, 8
x = torch.randn(B, L, D)
to_q = nn.Linear(D, D, bias=False)
q = to_q(x)
```

shape:

```text
x       : [2, 5, 8]
weight  : [8, 8]
q       : [2, 5, 8]
```

`nn.Linear(D, D)`는 마지막 차원 `D`를 변환하므로 앞의 `B`, `L`은 유지된다.

$$[B,L,D]\times[D,D]\rightarrow[B,L,D]$$

PyTorch의 `nn.Linear`는 개념적으로 다음과 같이 계산한다.

$$y=xW^T$$

- Linear의 weight는 처음에 초기화된 뒤 학습 과정에서 업데이트된다.
- 같은 Linear 가중치가 모든 token에 적용된다.

## 5. Token Embedding과 Position 정보

- Token Embedding: **이 token이 무엇인지 나타내는 의미 표현**
- Position 정보: **이 token이 sequence에서 어디에 있는지 나타내는 정보**

즉:

```text
Token Embedding = 명찰
Position        = 좌석 번호
```

두 정보를 더하려면 embedding dimension이 같아야 한다.

```text
Token Embedding    [B, L, D]
Position Embedding [B, L, D] 또는 [L, D]
----------------------------------------
결과               [B, L, D]
```

Position 정보를 더해도 sequence length `L`은 변하지 않는다.

## 6. Learned Position Embedding

Learned Position Embedding에서는 **각 위치 ID에 대응하는 embedding vector의 값이 학습된다.**

예:

```text
position 0 → [?, ?, ?, ?]
position 1 → [?, ?, ?, ?]
position 2 → [?, ?, ?, ?]
```

위치 번호 `0`, `1`, `2` 자체가 학습되는 것이 아니라 각 위치에 대응하는 벡터 값이 학습된다.

### Sinusoidal Position Encoding

여러 주기의 sine/cosine을 사용하여 각 위치를 서로 다른 패턴 조합으로 표현한다.

- 짧은 주기: 가까운 위치 차이를 세밀하게 표현
- 긴 주기: 먼 위치 관계를 표현

## 7. Encoder Block

Encoder Block의 핵심 작업:

```text
Self-Attention
→ Residual + LayerNorm
→ Position-wise FFN
→ Residual + LayerNorm
```

- Self-Attention: token 사이 정보를 교환
- FFN: 각 token의 표현을 개별적으로 가공
- Residual Connection: 원래 입력을 출력에 다시 더함
- LayerNorm: 값을 정규화하여 학습을 안정화

Encoder Layer 전후에는 보통 `[B, L, D]`가 유지된다.

## 8. Decoder-Only Block

Decoder-only Block은 일반적으로 다음 구조를 가진다.

```text
Masked Self-Attention
→ Residual + LayerNorm
→ Position-wise FFN
→ Residual + LayerNorm
```

Encoder와의 핵심 차이는 **Causal Mask**이다.

Causal Mask를 사용해 현재 token이 미래 token을 볼 수 없도록 한다.

## 9. Position-wise FFN

FFN은 Feed-Forward Network이다.

```python
self.ffn = nn.Sequential(
    nn.Linear(hidden_size, hidden_size * 2),
    nn.GELU(),
    nn.Linear(hidden_size * 2, hidden_size),
)
```

예를 들어:

```text
8 → 16 → 8
```

- `8 → 16`: 중간 차원을 늘려 더 다양한 특징 조합을 학습
- `GELU`: 비선형 변환
- `16 → 8`: 다시 원래 `hidden_size`로 복원

다시 원래 크기로 줄이는 이유:

- Residual Connection에서 원래 입력과 더하기 위해
- 다음 Layer가 같은 hidden size를 사용하도록 하기 위해

정답 표현:

**중간 차원을 늘려 더 다양한 특징 조합을 학습하고, 마지막에 다시 `hidden_size`로 줄여 Residual Connection과 다음 Layer의 입력 shape를 맞춘다.**

## 10. Position-wise의 의미

`Position-wise`는 **Position Embedding을 추가한다는 뜻이 아니다.**

sequence의 각 위치에 있는 token에 **동일한 FFN을 각각 독립적으로 적용한다**는 뜻이다.

```text
token 1 → FFN → 새로운 token 1 표현
token 2 → FFN → 새로운 token 2 표현
token 3 → FFN → 새로운 token 3 표현
```

- token끼리 직접 섞이지 않는다.
- 모든 위치가 같은 FFN weight를 사용한다.

핵심:

**Self-Attention은 token 사이 정보를 교환하고, Position-wise FFN은 각 token의 표현을 독립적으로 가공한다.**

## 11. Residual Connection

Residual Connection은 원래 입력 `x`를 연산 결과에 다시 더하는 연결이다.

```python
x = x + ffn(x)
```

따라서 `ffn(x)`의 출력과 원래 `x`의 shape가 같아야 한다.

## 12. LayerNorm 객체

```python
self.norm1 = nn.LayerNorm(hidden_size)
self.norm2 = nn.LayerNorm(hidden_size)
```

코드는 같지만 호출할 때마다 새로운 객체가 생성된다.

- `norm1`: 자기만의 `weight`, `bias`
- `norm2`: 자기만의 `weight`, `bias`

따라서 서로 독립적으로 학습된다.

`norm1`, `norm2`는 PyTorch에 정해진 이름이 아니라 개발자가 붙인 변수명이다.

## 13. Tensor Indexing

입력 shape가 다음과 같다고 하자.

```text
x.shape = [2, 4, 8]
          [B, L, D]
```

```python
changed[0, 1] += 10
```

은 다음과 같다.

```python
changed[0, 1, :] += 10
```

의미:

- `0`: 첫 번째 batch
- `1`: 두 번째 token
- `:`: 해당 token의 8차원 벡터 전체

즉 **첫 번째 batch의 두 번째 token 벡터 8개 값 모두에 10을 더한다.**

한 차원만 바꾸려면 세 번째 index까지 지정해야 한다.

```python
changed[0, 1, 3] += 10
```

## 14. KV Cache

KV Cache는 생성 과정에서 **이미 계산한 이전 token들의 Key와 Value를 저장해 재사용하는 메모리**이다.

```text
이전 token의 K/V → Cache에 저장
새 token 생성
→ 새 token의 K/V만 계산
→ 이전 K/V는 Cache에서 재사용
```

장점:

- 중복 계산 감소
- 생성 속도 향상

단점:

- sequence가 길어질수록 저장해야 하는 K/V가 증가
- 메모리 사용량 증가

즉:

**KV Cache는 생성 속도를 높이는 대신 메모리를 사용하는 구조이다.**

## 15. Attention Score와 KV Cache의 메모리 증가

Full Attention Score는 모든 token 쌍을 계산한다.

$$L\times L=L^2$$

따라서 sequence length를 2배로 늘리면:

```text
Attention Score → 4배
KV Cache        → 2배
```

KV Cache는 sequence length에 선형적으로 증가한다.

## 16. MHA와 GQA

- MHA: Query Head마다 K/V Head 사용
- GQA: 여러 Query Head가 K/V Head를 공유

KV Cache 크기는 `H_kv`에 비례한다.

예:

```text
MHA : H_q = 16, H_kv = 16
GQA : H_q = 16, H_kv = 4
```

이 경우:

```text
GQA KV Cache = MHA KV Cache의 1/4
```

즉 GQA는 K/V Head를 공유하여 KV Cache 메모리를 줄일 수 있다.

## 17. 핵심 암기

```text
Token Embedding → token이 무엇인가
Position        → sequence에서 어디에 있는가

Self-Attention  → token끼리 정보 교환
FFN             → 각 token 내부 표현 가공

Head            → 하나의 Attention 계산 단위
Multi-Head      → 여러 projection 관점에서 Attention 수행

Residual        → 원래 x를 다시 더함
LayerNorm       → 학습 안정화

FFN             → D → 확장 → D
Position-wise   → 각 위치의 token에 같은 FFN을 독립 적용

KV Cache        → 과거 K/V 저장 및 재사용
Attention Score → L²
KV Cache        → L에 비례
GQA             → H_kv를 줄여 KV Cache 절감
```