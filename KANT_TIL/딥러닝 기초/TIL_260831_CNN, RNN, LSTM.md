
# 2026-08-31 PyTorch CNN·RNN·LSTM 학습 정리

## 1. CNN 기본 흐름

```text
입력 이미지
-> Conv2d
-> ReLU
-> MaxPool2d
-> Conv2d
-> ReLU
-> MaxPool2d
-> Flatten
-> Classifier
-> Logits
```

| 단계 | 정의 | 사용하는 이유 |
|---|---|---|
| `Conv2d` | 작은 필터를 이동시키며 특징을 계산하는 층 | 선, 모서리, 무늬 등 지역적 특징 추출 |
| `ReLU` | 음수는 0, 양수는 그대로 반환하는 활성화 함수 | 비선형성을 추가하여 복잡한 특징 학습 |
| `MaxPool2d` | 작은 영역에서 최댓값만 선택하는 층 | 강한 특징을 남기며 공간 크기와 계산량 축소 |
| `Flatten` | Feature Map을 1차원 특징 벡터로 펼치는 연산 | `Linear` Classifier에 연결 |
| `Classifier` | 특징을 클래스별 점수로 변환하는 분류층 | 최종 클래스 판단 |
| `Logits` | Classifier가 출력한 정규화 전 클래스 점수 | 손실 계산 또는 클래스 선택에 사용 |

핵심 흐름:

```text
특징 추출
-> 비선형 변환
-> 특징 압축
-> 클래스 판단
```

## 2. Conv2d의 채널과 필터

```python
images = torch.randn(16, 3, 32, 32)
```

```text
(16, 3, 32, 32)
  |  |   |   |
  |  |   |   +-- Width
  |  |   +------ Height
  |  +---------- RGB Channel
  +------------- Batch Size
```

```python
nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1
)
```

```text
in_channels=3
-> 입력 RGB 채널 3개

out_channels=16
-> 서로 다른 필터 16개
-> Feature Map 16개 생성

kernel_size=3
-> 필터의 공간 크기 3x3

padding=1
-> 높이와 너비 유지
```

RGB 입력에서 필터 하나의 실제 가중치 크기는 `(3, 3, 3)`이다.

```text
R 채널의 3x3 계산
+ G 채널의 3x3 계산
+ B 채널의 3x3 계산
-> Feature Map의 값 하나
```

```text
필터 1개
-> Feature Map 1개
-> 출력 채널 1개

필터 16개
-> Feature Map 16개
-> 출력 채널 16개
```

다음 두 숫자 `16`은 의미가 다르다.

```text
images의 첫 번째 16
-> 이미지 수
-> Batch Size

Conv2d의 두 번째 16
-> 필터 수
-> 출력 채널 수
```

```python
images = torch.randn(16, 3, 32, 32)
labels = torch.randint(0, 10, size=(16,))
```

```text
이미지 16장
-> 정답 라벨 16개

출력 채널 16개
-> 모델이 추출할 특징 종류 16개

클래스 수 10개
-> 정답 번호 0~9
```

작은 CNN에서는 채널 수를 단계적으로 늘릴 수 있다.

```text
RGB 3채널
-> 필터 16개
-> Feature Map 16채널

Feature Map 16채널
-> 필터 32개
-> Feature Map 32채널
```

```python
nn.Conv2d(3, 16, kernel_size=3, padding=1)
nn.Conv2d(16, 32, kernel_size=3, padding=1)
```

앞부분에서는 비교적 단순한 특징을 적은 채널로 찾고, 뒤로 갈수록 채널을 늘려 다양한 복합 특징을 학습한다.

## 3. MaxPool2d

```python
nn.MaxPool2d(2)
```

`2`는 `kernel_size=2`, 즉 `2x2` 영역마다 최댓값 하나를 선택한다는 뜻이다.

`stride`를 생략하면 기본적으로 `kernel_size`와 같은 `2`가 사용된다.

```text
1  3
2  7

-> 최댓값 7
```

```text
입력 4x4
-> 네 개의 2x2 영역
-> 출력 2x2
```

```text
입력 32x32
-> 출력 16x16
```

배치 수와 채널 수는 유지하고 높이와 너비만 절반으로 줄인다.

```text
입력  : (16, 16, 32, 32)
출력  : (16, 16, 16, 16)
```

Conv에서는 중앙점을 기준으로 주변을 대칭적으로 보기 위해 `3x3` 같은 홀수 커널을 많이 사용한다.

Pooling에서는 공간 크기를 정확히 절반으로 줄이기 위해 `2x2, stride=2`를 많이 사용한다.

## 4. Feature Map부터 Logits까지

```python
self.features = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
```

`self.features`는 이미지에서 Feature Map을 추출하고 변환하는 부분이다.

```text
입력                (Batch, 3, 32, 32)
첫 번째 Conv 출력   (Batch, 16, 32, 32)
첫 번째 Pool 출력   (Batch, 16, 16, 16)
두 번째 Conv 출력   (Batch, 32, 16, 16)
두 번째 Pool 출력   (Batch, 32, 8, 8)
```

마지막 Feature Map을 펼치면:

```text
(Batch, 32, 8, 8)
-> Flatten
-> (Batch, 32 x 8 x 8)
-> (Batch, 2048)
```

```python
def forward(self, x):
    x = self.features(x)          # Feature Map
    x = self.flatten(x)           # 특징 벡터
    logits = self.classifier(x)   # 클래스별 점수

    return logits
```

Classifier에 들어가기 전의 `x`는 특징 벡터이고, `classifier(x)`의 출력이 logits이다.

```text
Feature Map
-> Flatten
-> 특징 벡터
-> Classifier
-> Logits
```

Classifier가 다음과 같다면:

```python
self.classifier = nn.Linear(2048, 10)
```

Shape은 다음처럼 변한다.

```text
입력  : (16, 2048)
출력  : (16, 10)
```

이미지 16장마다 클래스 10개의 점수를 출력한다.

```python
assert logits.shape == (images.size(0), 10)
```

`images.size(0)`은 `images`의 0번 차원인 Batch Size이다.

```text
images.shape = (16, 3, 32, 32)
images.size(0) = 16

(images.size(0), 10)
= (16, 10)
```

## 5. Transform

`transform`은 데이터가 모델에 들어가기 전에 적용하는 전처리 과정이다.

```text
원본 이미지
-> 크기 조정
-> Tensor 변환
-> 정규화
-> 모델 입력
```

```text
Train Transform
-> 랜덤 증강
-> Tensor 변환
-> 정규화

Validation/Test Transform
-> Tensor 변환
-> 정규화
```

`transform`은 CNN 모델 내부의 층이 아니라 데이터 전처리 과정이다.

## 6. 평가 시 model.eval()

검증·테스트·추론에서는 모델을 평가 모드로 전환한다.

```python
model.eval()

with torch.no_grad():
    output = model(x)
```

```text
model.eval()
-> Dropout 비활성화
-> BatchNorm이 학습 중 저장한 Running 통계 사용

torch.no_grad()
-> Gradient 계산 중지
```

`model.eval()`만으로 Gradient 계산까지 중지되는 것은 아니다.

## 7. CPU Thread와 메모리 확인

```python
torch.set_num_threads(1)
```

PyTorch의 CPU 연산에 사용할 Thread 수를 1개로 제한한다.

```text
주요 목적
-> 작은 실습의 실행 시간 비교
-> CPU 병렬 처리로 인한 변동 감소
```

GPU 연산에는 영향을 주지 않는다.

일반적인 작은 Python 코드에서는 메모리를 매번 확인하지 않는다. 딥러닝에서는 다음 항목들이 많은 메모리를 사용한다.

```text
모델 가중치
입력 Tensor
중간 계산 결과
Gradient
Optimizer 상태
```

GPU 메모리가 부족하면 `CUDA out of memory` 오류가 발생할 수 있다.

메모리 확인이 특히 필요한 경우:

```text
Batch Size 비교
모델 크기 비교
학습과 추론 비교
no_grad() 적용 전후 비교
CPU와 GPU 비교
모델 경량화
추론 최적화
```

## 8. VGG16과 ResNet

### VGG16

```text
Conv와 ReLU 반복
-> Max Pooling
-> Conv와 ReLU 반복
-> Classifier
```

구조가 단순하고 이해하기 쉽지만 파라미터와 계산량이 많다.

### ResNet

ResNet은 층을 건너뛰는 `Skip Connection`을 사용한다.

```text
일반 경로
x -> Conv -> ReLU -> Conv

Skip Connection
x ----------------------> 더하기
```

입력 정보를 뒤쪽에 직접 전달하여 깊은 모델에서도 Gradient 전달과 학습을 안정화한다.

## 9. MLP와 CNN 비교

MLP는 이미지를 Flatten하여 처리한다.

```text
(3, 32, 32)
-> Flatten
-> (3072,)
```

CNN은 작은 영역의 인접 픽셀 관계를 이용한다.

| 구분 | MLP | CNN |
|---|---|---|
| 입력 처리 | 이미지를 1차원으로 펼침 | 2차원 공간 구조 유지 |
| 연결 방식 | 모든 입력과 뉴런 연결 | 지역적 영역만 연결 |
| 특징 학습 | 공간 관계를 직접 활용하기 어려움 | 지역 특징 학습 |
| 가중치 | 위치마다 별도 가중치 | 같은 필터를 전체 위치에서 공유 |
| 파라미터 | 많아지기 쉬움 | 상대적으로 효율적 |
| 특징 구조 | 전체 픽셀을 한 번에 조합 | 단순 특징부터 복잡한 특징까지 계층화 |

CNN이 이미지에서 유리한 주된 이유:

```text
지역적 연결(Local Connectivity)
-> 가까운 픽셀의 관계를 이용

가중치 공유(Weight Sharing)
-> 같은 필터로 여러 위치에서 같은 특징 탐색

계층적 특징 학습
-> 선·모서리
-> 무늬·부분
-> 객체
```

CNN이 유리한 주된 이유는 MLP에서만 Gradient가 소실되기 때문이 아니다. CNN도 깊어지면 Gradient 소실이 발생할 수 있다.

## 10. Baseline 비교 시 주의사항

MLP와 CNN을 공정하게 비교하려면 모델 구조를 제외한 실험 조건을 같게 유지해야 한다.

```text
같은 Train·Validation·Test 데이터 분할
같은 전처리와 정규화
같은 Batch Size
같은 Epoch 수
같은 손실함수
같은 Optimizer와 Learning Rate
같은 Seed
같은 평가 지표
```

그래야 성능 차이가 모델 구조에서 발생했는지 판단할 수 있다.

비교 결과에 추가할 내용:

```text
Train·Validation Loss 학습 곡선
Train·Validation Accuracy 학습 곡선
두 모델의 최종 성능 비교
Train과 Validation 곡선 차이에 따른 과적합 판별
이미지에서 CNN이 MLP보다 적합한 이유 분석
```

## 11. Best Epoch 선택

Validation Loss가 가장 낮은 Epoch와 Validation Accuracy가 가장 높은 Epoch는 다를 수 있다.

| Epoch | Validation Loss | Validation Accuracy |
|---:|---:|---:|
| 1 | 0.60 | 70% |
| 2 | 0.45 | 78% |
| 3 | 0.35 | 84% |
| 4 | 0.30 | **87%** |
| 5 | **0.26** | 86% |
| 6 | 0.31 | 85% |

```text
Validation Loss 기준
-> Epoch 5 선택

Validation Accuracy 기준
-> Epoch 4 선택
```

학습 전에 Best Model의 기준을 정하고, 그 기준이 가장 좋은 Epoch를 선택한다.

```text
학습 안정성과 과적합 감시
-> 최저 Validation Loss

프로젝트 목표가 정확도
-> 최고 Validation Accuracy

불균형 분류
-> F1, AP, ROC-AUC 등 목표 지표
```

최종 Test 결과를 보고 Epoch를 선택하면 안 된다.

```text
Validation으로 모델 선택
-> 선택된 모델을 Test에서 한 번 평가
```

## 12. set_seed와 make_train_loader

```text
set_seed
-> 난수 생성기의 시작점 고정
-> 무작위 결과를 재현 가능하게 설정

make_train_loader
-> Train Dataset을 Batch로 묶음
-> shuffle=True이면 데이터 순서를 실제로 섞음
-> 학습 Loop에 Batch를 하나씩 제공
```

`set_seed()`가 데이터를 직접 섞는 것이 아니다.

```text
DataLoader
-> 데이터를 실제로 섞음

Seed
-> 무작위 순서를 다시 재현할 수 있게 함
```

## 13. RNN 기본 구조

RNN은 문자만 예측하는 모델이 아니라 순서가 있는 데이터를 처리하여 Hidden State를 만드는 구조이다.

```text
현재 입력 x_t
+ 이전 Hidden State h_(t-1)
-> 현재 Hidden State h_t
```

RNN을 시간축으로 펼치면:

```text
h_0 -> h_1 -> h_2 -> h_3
```

접힌 도식의 자기 자신을 향한 화살표는 값이 과거로 돌아간다는 뜻이 아니다.

```text
현재 Hidden State
-> 다음 시점의 같은 RNN Cell에 입력
```

각 시점에서는 같은 RNN Cell과 같은 가중치를 반복해서 사용한다.

## 14. RNN 입력 Shape와 batch_first

```python
x = torch.randn(4, 6, 3)

rnn = nn.RNN(
    input_size=3,
    hidden_size=8,
    batch_first=True
)
```

```text
x.shape = (4, 6, 3)
           |  |  |
           |  |  +-- 시점별 Feature 수
           |  +----- Sequence Length
           +-------- Batch Size
```

`batch_first=True`는 Batch Size를 지정하는 옵션이 아니다.

입력과 출력 Tensor에서 Batch 축을 첫 번째에 두겠다는 뜻이다.

```text
batch_first=True
-> (Batch, Sequence, Feature)
-> (4, 6, 3)

batch_first=False
-> (Sequence, Batch, Feature)
-> (6, 4, 3)
```

## 15. RNN의 output과 h_n

```python
output, h_n = rnn(x)
```

```text
output
-> 모든 시점의 Hidden State
-> [h_1, h_2, ..., h_6]

h_n
-> 마지막 시점의 Hidden State
-> h_6
```

현재 예시의 Shape:

```text
입력 x : (4, 6, 3)
output : (4, 6, 8)
h_n    : (1, 4, 8)
```

`output` 하나가 모든 정보를 합친 값이라는 뜻은 아니다.

각 시점에서 만들어진 Hidden State를 순서대로 담은 Tensor이다.

```text
output[:, 0, :] -> 첫 번째 시점의 Hidden State
output[:, 5, :] -> 여섯 번째 시점의 Hidden State
```

## 16. RNN의 작업 유형

RNN은 출력층을 어떻게 연결하는지에 따라 다른 작업에 사용한다.

### 문장 전체 분류

```text
문장 전체
-> RNN
-> 마지막 Hidden State
-> Classifier
-> 클래스 하나
```

활용 예:

```text
감성 분류
-> 긍정 / 부정

스팸 분류
-> 정상 / 스팸

의도 분류
-> 질문 / 주문 / 환불 / 예약

문서 주제 분류
-> 정치 / 경제 / 스포츠 / 기술

시계열 상태 분류
-> 정상 / 이상
```

`RNNClassifier`는 RNN이 만든 특징을 분류용 `Linear` 층에 넣기 때문에 붙인 이름이다.

```python
self.rnn = nn.RNN(...)
self.classifier = nn.Linear(hidden_size, num_classes)
```

### 토큰별 분류

각 시점의 Hidden State를 각각 Classifier에 넣는다.

```text
입력 토큰: 나는     학교에    간다
품사 라벨: 대명사   명사      동사
```

```text
나는
-> h_1
-> Classifier
-> 대명사

학교에
-> h_2
-> Classifier
-> 명사

간다
-> h_3
-> Classifier
-> 동사
```

품사 태깅과 개체명 인식 등에 사용한다.

### 다음 단어 예측

```text
입력: 나는
-> 예측: 학교에

입력: 학교에
-> 예측: 간다
```

`나는 학교에`를 입력하여 `간다` 하나만 예측한다면 마지막 Hidden State를 사용한다.

```text
마지막 Hidden State
-> Vocabulary 크기의 Linear 층
-> 단어별 Logits
-> “간다” 선택
```

다음 문자나 단어를 예측하는 것도 Vocabulary 후보 중 하나를 고르는 분류 문제로 볼 수 있다.

## 17. Padding과 실제 마지막 토큰

길이가 다른 문장들을 하나의 Batch Tensor로 만들려면 길이를 맞춰야 한다.

```text
문장 1: 나는 학교에 간다
문장 2: 나는 잔다   PAD
```

Tensor는 같은 dtype의 숫자로 구성된 직사각형 배열이다.

따라서 `null`을 넣는 대신 `0` 같은 특별한 PAD Token 번호를 사용한다.

Padding이 있을 때 다음 코드는 짧은 문장의 PAD 위치를 가져올 수 있다.

```python
output[:, -1, :]
```

PAD를 뒤에서 하나씩 검사하는 대신 문장별 실제 길이를 함께 저장한다.

```text
lengths = [3, 2]

문장 1의 마지막 실제 index
-> 3 - 1
-> 2

문장 2의 마지막 실제 index
-> 2 - 1
-> 1
```

PyTorch에서는 `pack_padded_sequence()`로 PAD를 계산 대상에서 제외할 수 있다.

```python
packed = pack_padded_sequence(
    embedded,
    lengths,
    batch_first=True,
    enforce_sorted=False
)

output, h_n = self.rnn(packed)
```

이때 `h_n`은 PAD 위치가 아니라 각 문장의 실제 마지막 토큰까지 계산한 Hidden State이다.

## 18. tanh와 Sigmoid

| 함수 | 출력 범위 | 입력이 0일 때 |
|---|---:|---:|
| Sigmoid | 0 ~ 1 | 0.5 |
| tanh | -1 ~ 1 | 0 |

둘 다 S자 형태지만 `tanh`는 0을 중심으로 음수와 양수를 모두 출력한다.

```text
Sigmoid
-> 출력 범위 0~1

tanh
-> 출력 범위 -1~1
```

`tan`은 일정 주기로 반복되고 특정 지점에서 발산하는 삼각함수이다. RNN에서 사용하는 것은 `tan`이 아니라 `tanh`이다.

## 19. Hidden State의 내용

Hidden State는 Hidden Layer 자체가 아니다.

현재 시점까지 입력된 Sequence의 특징을 요약한 숫자 벡터이다.

```text
입력:
“나는 학교에”

Hidden State 예시:
[0.72, -0.15, 0.43, ..., 0.21]
```

Hidden State에는 다음과 같은 정보가 분산되어 표현될 수 있다.

```text
이전에 나온 단어
현재 문장의 주제
문법적 흐름
긍정·부정 경향
다음 예측에 필요한 문맥
```

특정 숫자 하나가 항상 특정 문법 의미를 담당하는 것은 아니다. 여러 숫자가 함께 문맥을 표현한다.

## 20. LSTM 등장 배경

기존 RNN은 Sequence가 길어질수록 앞부분의 정보를 뒤까지 전달하기 어렵다.

```text
긴 Sequence
-> Hidden State 반복 계산
-> Gradient 소실 또는 폭발
-> 장기 의존성 학습 어려움
```

LSTM은 장기 기억 통로인 Cell State와 정보 흐름을 조절하는 Gate를 추가하여 이 문제를 완화한다.

```text
Hidden State h_t
-> 현재 시점까지의 정보를 요약한 현재 출력 상태

Cell State c_t
-> 긴 Sequence에서도 정보를 유지하는 장기 기억 통로
```

각 시점의 계산:

```text
현재 입력 x_t
+ 이전 Hidden State h_(t-1)
+ 이전 Cell State c_(t-1)
-> 현재 h_t와 c_t 생성
```

## 21. LSTM Gate

| 구성 | 역할 | 핵심 질문 |
|---|---|---|
| Forget Gate | 기존 Cell State에서 불필요한 정보 삭제 | 무엇을 잊을 것인가? |
| Input Gate | 새로운 후보 정보를 얼마나 저장할지 결정 | 얼마나 기억할 것인가? |
| Candidate | 현재 입력으로 새로운 기억 후보 생성 | 새로 기억할 내용은 무엇인가? |
| Output Gate | Cell State 중 Hidden State로 출력할 정보 결정 | 무엇을 현재 출력으로 사용할 것인가? |

Candidate는 엄밀히 말하면 정보량을 조절하는 Gate가 아니다.

Cell State에 추가할 새로운 기억 후보이다.

```text
기존 Cell State
-> Forget Gate로 불필요한 기억 삭제
-> Input Gate x Candidate로 새 기억 추가
-> 새로운 Cell State 생성
-> Output Gate를 거쳐 Hidden State 생성
```

## 22. LSTM Gate 문장 예시

```text
“나는 프랑스에서 태어나 여러 나라를 여행하고 지금은 서울에 살지만,
나의 모국어는 프랑스어다.”
```

### `프랑스에서 태어나`

```text
Candidate
-> [출생 국가: 프랑스] 기억 후보 생성

Input Gate
-> 중요한 정보로 판단하여 Cell State에 저장
```

### `여러 나라를 여행하고`

```text
Forget Gate
-> 불필요한 여행 세부 정보 삭제
-> [출생 국가: 프랑스] 정보 유지
```

### `지금은 서울에 살지만`

```text
Candidate
-> [현재 거주지: 서울] 정보 생성

Input Gate
-> 현재 거주지 정보 저장

Cell State
-> 출생 국가와 현재 거주지 정보를 구분하여 유지
```

### `나의 모국어는`

```text
Output Gate
-> 모국어 예측에 필요한 프랑스 정보를 출력
```

### `프랑스어다`

```text
Hidden State
-> 프랑스 관련 문맥 포함

출력층
-> 프랑스어의 Logit을 높게 계산
```

이 과정은 Gate의 역할을 이해하기 위한 단순화된 예시이다.

실제 내부 값은 사람이 읽는 문장 형태가 아니라 숫자 벡터이다.

## 23. LSTM에서 Transformer로 발전한 배경

LSTM은 RNN의 장기 의존성 문제를 완화했지만 두 가지 한계가 남아 있다.

```text
1. 순차 처리
-> 이전 Hidden State가 있어야 다음 토큰 계산 가능
-> 모든 토큰을 동시에 처리하기 어려움

2. 장거리 관계
-> 먼 토큰의 정보가 여러 시점을 거쳐 전달됨
-> 매우 긴 Sequence의 관계 학습에 여전히 한계
```

Transformer는 순환 구조 대신 Self-Attention을 사용한다.

```text
LSTM
-> 토큰을 순서대로 처리
-> 이전 상태를 거쳐 정보 전달

Transformer
-> 여러 토큰을 병렬로 처리
-> 각 토큰이 다른 모든 토큰을 직접 참고
```

```text
LSTM
프랑스
-> 여러 중간 시점
-> 모국어

Transformer
프랑스
<-> Self-Attention
<-> 모국어
```

Transformer는 LSTM의 순차 처리에 따른 병렬화 한계와 장거리 의존성 문제를 개선하기 위해 등장했다.

## 24. 면접 코멘트

튜터 코멘트:

```text
RNN·CNN·LSTM의 세부 구조 설명은 면접 질문 가능성이 낮음

Transformer 구조는 면접에서 질문할 수 있음

Transformer를 수학 수식까지 설명할 필요는 없음

핵심 구성요소와 데이터 흐름을 말로 설명할 수 있도록 준비
```

## 25. 오늘의 핵심 요약

```text
CNN
-> 지역적 연결과 가중치 공유로 이미지 특징을 효율적으로 추출

Feature Map
-> Conv 필터가 입력에서 찾아낸 특징 지도

MaxPool2d(2)
-> 각 2x2 영역의 최댓값을 남겨 높이와 너비를 절반으로 축소

Classifier
-> Flatten된 특징을 클래스별 Logits로 변환

RNN
-> 이전 Hidden State를 다음 시점에 전달하며 Sequence 처리

RNN output
-> 모든 시점의 Hidden State를 순서대로 저장

h_n
-> 마지막 시점의 Hidden State

LSTM
-> Cell State와 Gate로 RNN의 장기 의존성 문제 완화

Transformer
-> Self-Attention으로 토큰 간 관계를 직접 계산하고 병렬 처리
```