
## 1. RNN · LSTM · Transformer

- RNN은 현재 입력 `x_t`와 이전 hidden state `h_{t-1}`를 이용해 새로운 `h_t`를 계산한다.

$$h_t=\tanh(W_xx_t+W_hh_{t-1}+b)$$

- 시간축 의존성: 현재 계산이 이전 hidden state에 의존하므로 한 sequence 내부는 순차 계산한다.
- RNN의 양 끝 위치 정보 전달 경로: `L-1`
- `output [B,L,H]`: 모든 시점 hidden state
- `h_n [layers×directions,B,H]`: 각 layer·direction의 마지막 hidden state
- Gradient Clipping: Exploding Gradient는 완화하지만 Vanishing Gradient·장기 의존성·순차 처리 문제는 해결하지 못한다.
- LSTM: cell state와 gate로 장기 의존성과 gradient 문제를 완화하지만 완전히 해결하지는 못한다.
- BiRNN/BiLSTM: 전체 sequence가 주어진 상태에서 앞뒤 문맥을 함께 활용한다. 미래 입력이 없는 실시간 예측에는 backward 방향을 사용할 수 없다.
- Self-Attention은 먼 token을 직접 연결해 maximum path length가 `O(1)`이지만 모든 token 쌍을 계산해 `L²` 비용이 발생한다.
- Transformer 학습은 정답 sequence가 모두 주어져 병렬 계산 가능하지만, 생성은 이전 생성 token에 의존하므로 순차적이다.

## 2. 생성형 모델 전체 흐름

```text
Raw Text
→ Tokenizer
→ input_ids / attention_mask
→ Model
→ Model Head
→ logits
→ token/class 선택
→ 결과
```

- Forward: 현재 입력의 logits를 한 번 계산
- `generate()`: token 선택 → append → forward를 반복
- 분류 logits `[B,C]`: 샘플 수 × 클래스 수
- 생성 logits `[B,L,V]`: batch × sequence × vocabulary
- Accuracy는 정답 비율, Loss는 학습을 위해 예측과 정답 차이를 수치화한 값이다.
- 생성 문장의 자연스러움과 사실성은 별개다.
- 사실성은 claim을 외부 데이터·DB·API·검색 근거와 대조해 별도로 검증한다.
- 마지막 checkpoint가 항상 best checkpoint는 아니며 validation metric으로 선택한다.

## 3. Tokenizer · Token · Vocabulary

> 토큰 = 모델이 처리하는 최소 입력 단위

- Token은 반드시 단어인 것은 아니다. (토큰이 단어일 수도 있지만, subword·문자 등일 수도 있다)
- Word: `playing`
- Subword: `play + ing`
- Character: `p + l + a + y + i + n + g`
- Subword는 word보다 작은 조각을 재사용해 OOV를 줄이면서 character보다 sequence를 덜 늘린다.
- Token ID: tokenizer vocabulary에서 token에 부여된 식별자
- 같은 문자열도 tokenizer마다 vocabulary·분할 규칙이 달라 Token ID가 다를 수 있다.
- Model과 tokenizer는 같은 Token ID를 학습 때와 동일한 token으로 해석해야 하므로 같은 checkpoint를 사용한다.
- 예: 학습 당시 `125 = 학교`인데 다른 tokenizer에서 `125 = 사과`라면 모델이 입력을 잘못 해석한다.

## 4. Vocabulary Size

- Vocabulary가 크다 = 사전에 등록된 token 종류가 많다.
- Vocabulary가 크면 긴 문자열을 하나의 token으로 표현할 가능성이 커져 sequence가 짧아질 수 있다.
- 대신 Embedding과 LM Head 파라미터가 증가한다.
- Embedding: `Token ID → hidden vector`
- LM Head: `hidden vector → vocabulary 전체 logits`
- Vocabulary가 작으면 희귀 문자열을 더 작은 subword로 표현하므로 sequence가 길어질 수 있다.
- Vocabulary size만으로 tokenizer 품질을 평가하지 않고 algorithm·corpus·정규화 규칙도 함께 본다.

## 5. Special Token

- `[CLS]`: 문장 전체를 대표하는 위치
- `[SEP]`: 서로 다른 입력 segment의 경계를 표시
- `[PAD]`: 길이를 맞추기 위한 채움 token
- `[MASK]`: BERT의 Masked Language Modeling에 사용
- `[UNK]`: vocabulary에서 표현하지 못한 입력

`[SEP]`의 핵심 사용 목적:

- 두 문장의 관계 판단
- 질문과 지문/답변 문맥 구분
- 문장 쌍 분류
- Entailment·Similarity 판단

Special token과 ID는 하드코딩하지 않고 tokenizer 객체에서 확인한다.

## 6. Padding · Truncation · max_length

- Padding = 짧은 입력을 채워 batch shape를 맞춤
- Truncation = 너무 긴 입력을 자름
- `max_length` = 최대 길이 기준
- `max_length`만 지정한다고 자동으로 잘리는 것이 아니므로 `truncation=True`를 함께 사용한다.
- 동적 Padding: 현재 batch의 최장 길이까지만 PAD를 추가
- Dataset에서는 가변 길이를 유지하고 Collator에서 동적 padding하는 것이 효율적이다.
- PAD 위치는 `attention_mask=0`이어야 모델이 해당 위치를 참조하지 않는다.
- `max_length`는 token 길이 분포·truncation 비율·성능·메모리를 함께 보고 정한다.

## 7. Batch · Collator · Dataset

- `collate`: 여러 샘플을 취합해 하나의 batch로 묶는 것
- `DataCollatorWithPadding`: 현재 batch 최장 길이에 맞춰 동적 padding
- `DatasetDict.map(batched=True)`: train·validation·test에 동일한 전처리를 batch 단위로 적용
- Split 경계를 먼저 고정한 후 같은 전처리를 적용해 데이터 누수를 방지한다.
- `input_ids [6,14]`에서 `6`은 batch 크기, `14`는 동적 padding 후 현재 batch의 sequence 길이
- `return_tensors="pt"`를 사용하면 PyTorch Tensor가 되어 `.shape`, `.to(device)` 등을 바로 사용할 수 있다.
- `BatchEncoding`의 key는 모델마다 다를 수 있으므로 `token_type_ids` 등이 항상 있다고 가정하지 않는다.

## 8. Label ID와 Token ID

- Label ID: task에서 정한 분류 class 번호
- Token ID: tokenizer vocabulary에서 정한 token 번호
- 예:
  - `"economy" → label ID 0`
  - `"[PAD]" → token ID 0`
- 둘 다 정수지만 서로 다른 namespace이므로 섞어 해석하면 안 된다.
- `label2id`: 사람이 읽는 label → class ID
- `id2label`: class ID → 사람이 읽는 label
- Dataset의 `label`이 모델이 기대하는 `labels`로 정상 전달되어야 loss를 계산할 수 있다.

## 9. Train · Validation · Test · Inference

- Train: parameter 학습
- Validation: 모델·hyperparameter·best checkpoint 선택
- Test: 선택이 끝난 모델의 최종 성능 확인
- Inference: 새로운 입력에 실제 예측 수행
- Validation과 Test를 분리해 선택에 사용한 데이터가 최종 평가에 영향을 주는 낙관적 편향을 줄인다.

## 10. 재현성과 오류 분석

- Model과 tokenizer checkpoint를 함께 기록·저장한다.
- Tokenizer가 달라지면 같은 문자열도 다른 `input_ids`가 될 수 있다.
- `remove_columns` 전에 error analysis에 필요한 `id`, `text` 보존 여부를 결정한다.
- Split 간 중복이 있으면 train에서 본 데이터를 test에서 다시 평가해 일반화 성능이 부풀려질 수 있다.
- 실행 환경·설정·checkpoint·metric·산출물 경로를 함께 기록해야 실험을 재현할 수 있다.