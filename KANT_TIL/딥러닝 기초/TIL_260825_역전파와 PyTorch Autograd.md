---
date: 2026-08-25
tags:
  - PyTorch
  - Autograd
  - Gradient
  - Evaluation
---
# PyTorch 학습·평가 흐름

## 1. 예측값과 Loss

```python
x = 2
w = 3
b = 1

z = x * w + b  # 7
loss = z ** 2  # 49
```

- 예측 함수: `f(x) = xw + b`
- 예측값: `z = 7`
- `loss = z ** 2`는 목표값을 `0`으로 가정한 예제

$$L=(z-0)^2=z^2$$

일반적인 회귀의 제곱오차는 정답 `y`와 비교함.

$$L=(z-y)^2$$

```text
입력 x -> 예측값 z=xw+b -> 정답 y와 비교 -> Loss
```

## 2. Chain Rule

계산 흐름:

```text
w -> z=xw+b -> L=z²
```

각 구간의 변화율:

$$\frac{\partial L}{\partial z}=2z=14$$

$$\frac{\partial z}{\partial w}=x=2$$

연결된 전체 변화율:

$$\frac{\partial L}{\partial w}=\frac{\partial L}{\partial z}\times\frac{\partial z}{\partial w}=14\times2=28$$

변화율을 곱하는 이유:

```text
w가 0.001 증가
-> z가 2 × 0.001 = 0.002 증가
-> Loss가 14 × 0.002 = 0.028 증가
```

따라서:

$$\frac{\Delta L}{\Delta w}=\frac{0.028}{0.001}=28$$

> `w -> z -> Loss`처럼 변화가 연속해서 전달되므로 각 구간의 변화율을 곱함.

## 3. Gradient의 의미

$$\frac{\partial L}{\partial w}=28$$

Gradient `28`에서 알 수 있는 것:

| 구분 | 의미 |
|---|---|
| 부호 `+` | `w`가 증가하면 Loss 증가 |
| 반대 방향 | Loss를 줄이려면 `w` 감소 |
| 크기 `28` | 현재 위치에서 Loss가 `w` 변화에 민감한 정도 |

```text
w가 0.001 증가
-> Loss가 약 28 × 0.001 = 0.028 증가
```

하지만 `28`만 보고 무조건 큰 Gradient라고 판단할 수는 없음.

- 입력값의 크기
- Loss의 단위와 계산 방식
- 파라미터 크기
- 데이터의 평균·합산 방식
- 모델 구조

등에 따라 Gradient의 크기가 달라짐.

## 4. Gradient, Learning Rate, `optimizer.step()`

Gradient는 방향과 민감도를 알려주고, Learning Rate는 실제 이동 크기를 결정함.

$$w_{\mathrm{new}}=w_{\mathrm{old}}-\mathrm{lr}\times\frac{\partial L}{\partial w}$$

Learning Rate가 `0.01`이면:

$$w_{\mathrm{new}}=3-(0.01\times28)=2.72$$

`b`의 Gradient:

$$\frac{\partial L}{\partial b}=\frac{\partial L}{\partial z}\times\frac{\partial z}{\partial b}=14\times1=14$$

$$b_{\mathrm{new}}=1-(0.01\times14)=0.86$$

수정된 값으로 다시 예측하면:

$$z=(2\times2.72)+0.86=6.3$$

$$L=6.3^2=39.69$$

```text
수정 전 Loss: 49
수정 후 Loss: 39.69
```

전체 역할:

```text
model(x)         -> 예측값 계산
loss             -> 예측 오차 계산
loss.backward()  -> Gradient 계산
Learning Rate    -> 이동 크기 조절
optimizer.step() -> 모델 파라미터 수정
```

## 5. `requires_grad=True`

```python
x = torch.tensor(2.0, requires_grad=True)
```

`requires_grad=True`는 이 Tensor가 사용된 연산을 기록하여, 나중에 `backward()`로 Gradient를 계산할 수 있게 한다는 의미임.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

y.backward()

print(x.grad)  # tensor(4.)
```

계산식:

$$y=x^2$$

$$\frac{dy}{dx}=2x=4$$

```text
requires_grad=True
-> 연산 과정 기록
-> backward()
-> Gradient 계산
-> .grad에 저장
```

모델의 가중치와 편향은 학습 대상이므로 기본적으로 `requires_grad=True`임.

```python
model.weight.requires_grad  # True
model.bias.requires_grad    # True
```

`requires_grad=False`인 파라미터는 Gradient를 계산하지 않으므로 학습 중 변경되지 않음.

## 6. `loss.backward()`와 `.grad`

```python
x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

z = x * w + b
loss = z ** 2

loss.backward()
```

역전파 흐름:

```text
Loss=49
   ↓  dL/dz=14
z=7
   ├-> dL/dw=14×2=28 -> w.grad=28
   └-> dL/db=14×1=14 -> b.grad=14
```

`backward()` 전후 비교:

| 시점 | `w.grad` | `b.grad` |
|---|---:|---:|
| `backward()` 전 | `None` | `None` |
| `backward()` 후 | `28` | `14` |

```python
print(w.grad)  # tensor(28.)
print(b.grad)  # tensor(14.)
```

`loss.backward()`는 파라미터를 수정하지 않고 `.grad`만 채움. 실제 수정은 `optimizer.step()`이 수행함.

Gradient는 누적되므로 반복 학습 전에 초기화해야 함.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 7. Parameter와 Gradient의 Shape

각 Parameter 원소마다 Gradient가 하나씩 계산되므로 두 Shape는 같음.

```python
layer = nn.Linear(3, 2)

x = torch.tensor([[1.0, 2.0, 3.0]])
loss = layer(x).sum()
loss.backward()
```

| 구분 | Parameter Shape | Gradient Shape |
|---|---|---|
| Weight | `(2, 3)` | `(2, 3)` |
| Bias | `(2,)` | `(2,)` |

```python
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        print(
            name,
            parameter.shape,
            parameter.grad.shape,
            parameter.shape == parameter.grad.shape
        )
```

`backward()` 전에는 `.grad=None`이므로 `.grad.shape`를 확인할 수 없음.

## 8. `detach()`와 `item()`

| 구분 | `detach()` | `item()` |
|---|---|---|
| 반환값 | Tensor | Python 숫자 |
| 여러 원소 | 가능 | 불가능 |
| Gradient 연결 | 끊음 | Tensor 밖으로 값을 꺼냄 |
| 용도 | 저장·시각화·NumPy 변환 | Loss 출력·기록 |

### `detach()`

```python
pred = model(x)
pred_for_check = pred.detach()
```

```text
pred           -> 모델과 계산 그래프 연결 O
pred.detach()  -> 계산 그래프 연결 X
```

`detach()`는 모델을 영구적으로 끊지 않고, 현재 출력으로 만든 Tensor만 계산 그래프에서 분리함.

```python
pred = model(x)
loss = criterion(pred, y)

pred_for_check = pred.detach()
loss.backward()  # 정상적으로 역전파
```

Loss 계산 전에 분리하면 모델까지 Gradient가 전달되지 않음.

```python
pred = model(x).detach()
loss = criterion(pred, y)
# 모델 파라미터까지 Gradient가 전달되지 않음
```

`model(x)`를 다시 호출하면 새로운 계산 그래프가 만들어짐.

### `item()`

원소가 하나인 Tensor를 Python 숫자로 변환함.

```python
loss_value = loss.item()
print(loss_value)
```

```text
detach() -> Tensor 형태 유지
item()   -> Python 숫자로 변환
```

## 9. 평가와 `torch.no_grad()`

평가에서는 가중치를 수정하지 않으므로 계산 그래프가 필요 없음.

```python
model.eval()

with torch.no_grad():
    prediction = model(x_val)
    val_loss = criterion(prediction, y_val)
```

`torch.no_grad()`의 효과:

- 계산 그래프 생성 중단
- 역전파용 중간값 저장 중단
- 메모리 사용량 감소
- 평가 속도 향상

### `model.eval()`과의 차이

| 코드 | 역할 |
|---|---|
| `model.eval()` | Dropout·BatchNorm을 평가 방식으로 변경 |
| `torch.no_grad()` | Gradient 기록과 계산 그래프 생성을 중단 |

둘은 역할이 다르므로 평가할 때 함께 사용함.

```python
model.eval()

with torch.no_grad():
    prediction = model(x_val)
```

다시 학습할 때:

```python
model.train()
```

## 10. Prediction과 Metric

- Prediction: 모델이 입력을 보고 만든 예측 결과
- Metric: Prediction과 정답을 비교한 평가 점수

```text
입력
-> 모델 원시 출력
-> Prediction 생성
-> 실제 정답과 비교
-> Metric 계산
```

| 문제 | Prediction | Metric 예시 |
|---|---|---|
| 회귀 | 연속값 | MAE, MSE, RMSE |
| 이진 분류 | 0 또는 1 | Accuracy, Precision, Recall, F1 |
| 다중 분류 | 클래스 번호 | Accuracy, Macro F1 |

```python
model.eval()

with torch.no_grad():
    logits = model(x_val)
    prediction = logits.argmax(dim=1)
    accuracy = (prediction == y_val).float().mean()
```

주의:

```text
Accuracy·F1
-> 최종 클래스 Prediction 사용

ROC-AUC·AP
-> Threshold 적용 전 확률·점수 사용
```

## 11. Dropout과 BatchNorm

| 구분 | Dropout | BatchNorm |
|---|---|---|
| 목적 | 과적합 방지 | 학습 안정화 |
| 방법 | 일부 출력을 무작위로 0 처리 | 중간 출력의 분포 조정 |
| 학습 모드 | 무작위 제거 적용 | 현재 Batch 통계 사용 |
| 평가 모드 | 모든 뉴런 사용 | 저장된 통계 사용 |

### Dropout

```python
nn.Dropout(p=0.5)
```

학습 중 일부 뉴런 출력을 무작위로 제거하여 특정 뉴런에 지나치게 의존하는 것을 방지함.

```text
model.train() -> Dropout 적용
model.eval()  -> Dropout 중단
```

### BatchNorm

현재 Mini-batch의 평균과 분산으로 중간 출력을 정규화한 후, 학습 가능한 `γ`, `β`로 다시 조정함.

$$\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$$

$$y=\gamma\hat{x}+\beta$$

```text
model.train()
-> 현재 Batch의 평균·분산 사용
-> running mean·variance 갱신

model.eval()
-> 학습 중 저장한 running mean·variance 사용
```

일반적인 배치 예:

```python
nn.Sequential(
    nn.Linear(20, 10),
    nn.BatchNorm1d(10),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(10, 1)
)
```

## 12. Python `with`문

Python의 `with`문은 특정 작업 환경을 적용하고, 블록이 끝나면 자동으로 정리하거나 원래 상태로 복구하는 문법임.

```python
with torch.no_grad():
    prediction = model(x_val)
```

```text
Gradient 기록 중단
-> with 블록 실행
-> 블록 종료
-> 기존 Gradient 기록 상태 복구
```

직접 상태를 관리하면 다음과 비슷함.

```python
torch.set_grad_enabled(False)

try:
    prediction = model(x_val)
finally:
    torch.set_grad_enabled(True)
```

`with`문은 오류가 발생해도 정리 작업과 상태 복구를 자동으로 처리함.

## 13. Python `with`와 SQL `WITH`

이름만 같고 역할은 다름.

| 구분 | Python `with` | SQL `WITH` |
|---|---|---|
| 개념 | Context Manager | CTE |
| 목적 | 실행 환경 설정·정리 | 임시 조회 결과 정의 |
| 범위 | 들여쓰기된 코드 블록 | 하나의 SQL문 |
| 종료 | 상태 복구·자원 정리 | 쿼리 종료 후 결과 소멸 |

SQL 예:

```sql
WITH high_salary AS (
    SELECT employee_id, salary
    FROM employee
    WHERE salary >= 5000
)
SELECT *
FROM high_salary;
```

```text
Python with
-> 특정 환경을 잠시 적용하고 자동 해제

SQL WITH
-> 조회 결과에 임시 이름을 붙여 쿼리에서 사용
```

## 14. 학습과 평가의 전체 흐름

```python
# 학습
model.train()

optimizer.zero_grad()
prediction = model(x_train)
loss = criterion(prediction, y_train)
loss.backward()
optimizer.step()

# 평가
model.eval()

with torch.no_grad():
    prediction = model(x_val)
    val_loss = criterion(prediction, y_val)
    metric = calculate_metric(prediction, y_val)
```

```text
[학습]
model.train()
-> 순전파
-> Loss
-> backward()
-> .grad
-> optimizer.step()

[평가]
model.eval()
-> no_grad()
-> Prediction
-> Validation Loss·Metric
```