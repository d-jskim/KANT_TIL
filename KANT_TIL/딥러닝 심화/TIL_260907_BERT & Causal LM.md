
## Embedding의 관계 표현

- Embedding 공간에서는 단어뿐 아니라 단어 사이의 관계도 벡터 방향으로 어느 정도 표현될 수 있다.
- `Paris - France ≈ Berlin - Germany`는 두 벡터가 `수도 - 국가`라는 유사한 관계를 표현한다는 의미이다.

## Causal LM

- Causal LM은 이전 token들만 보고 다음 token을 예측한다.
- 미래 token은 예측해야 할 정답이므로 Causal Mask를 사용해 볼 수 없도록 한다.
- GPT 계열 생성형 모델에서 사용하는 대표적인 방식이다.

## BERT와 Masked Language Modeling

- BERT는 대표적으로 다음 token 예측이 아니라 Masked Language Modeling(MLM)으로 사전학습한다.
- 문장의 일부 token을 `[MASK]`로 가리고 앞뒤 문맥을 이용해 원래 token을 예측한다.
- `[MASK]`는 정답 token을 숨겨 주변 문맥만으로 복원하도록 하기 위해 사용한다.
- 문장 자체에서 정답 token을 만들어 학습하므로 자기지도학습에 해당한다.

## Causal LM과 BERT 비교

| 구분 | Causal LM | BERT MLM |
|---|---|---|
| 예측 대상 | 다음 token | 가려진 token |
| 참조 문맥 | 이전 token | 앞뒤 token |
| 대표 모델 | GPT | BERT |
| Mask 목적 | 미래 정답 차단 | 정답 token 숨기기 |