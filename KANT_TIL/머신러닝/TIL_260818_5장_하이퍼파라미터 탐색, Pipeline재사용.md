# 5장 1강: CV 기반 하이퍼파라미터 탐색과 공정한 후보 선정

## 학습 목표
1. 하이퍼파라미터 후보를 단일 validation 분할의 점수가 아니라 **동일한 교차검증**으로 비교해야 하는 이유 설명
2. Grid Search와 Random Search를 같은 후보 수 · fold 수· 평가지표로 실행하고 CV fit 수를 비교 가능
   -> `cv_results`
3. CV 평균 · fold 간 변동을 우선하고 동률이면 트리 수와 깊이를 보조 기준으로 적용해 후보를 먼저 고정한 뒤, 봉인 test를 한 번 평가 가능

### 파라미터
모델이 학습하는 과정에서 배우는 값
### 하이퍼파라미터
사람이 정하는 설정

`공정한 후보` : 하이퍼파라미터 조합을 **동일한 조건에서 비교할 수 있는 후보**
- 같은 학습 데이터
- 같은 검증 데이터 또는 같은 교차검증 분할
- 같은 평가 지표
- 같은 전처리 조건
-> 특정 후보에 유리하거나 불리한 평가 조건을 주지 않고 동일 기준으로 비교한다

**모델 내부 학습** → 파라미터를 찾음  
**GridSearchCV 외부 탐색** → 하이퍼파라미터를 찾음
```
max_depth 후보 = [3, 5, 10]

3  → 모델 학습 → CV 평가
5  → 모델 학습 → CV 평가
10 → 모델 학습 → CV 평가

        ↓
점수 비교
        ↓
best_params_ = 가장 좋은 하이퍼파라미터
best_index_  = 그 후보의 위치
```


- (블로그 대신에) 공식문서 보는 습관
-  ChatGPT에 물어볼 때도 공식 문서 기반으로 설명 요청

#### 실습 코드 질문
- cv를 StratifedKFold로 설정하고 이 cv를 모델과 함께 GridSearchCV와 RandomizedSearchCV에 전달하는데, 3개 다 CV가 붙어있어서 헷갈림. cv는 데이터 분할 방법이고, 나머지 2개는 최적의 하이퍼파라미터를 찾는 것으로 알고 있는데 명확하게 설명이 안 됨.
```
하이퍼파라미터 후보 생성
        ↓
각 후보마다
        ↓
StratifiedKFold 방식으로 4분할
        ↓
4번 학습·검증
        ↓
평균 CV 성능 계산
        ↓
후보끼리 비교
        ↓
최적 하이퍼파라미터 선택
```

```
StratifiedKFold
"어떻게 나눌까?"
        ↓
        ↓ cv로 전달
        ↓
GridSearchCV / RandomizedSearchCV
"그렇게 나눠서 각 하이퍼파라미터를 평가해보자"
        ↓
최적 하이퍼파라미터 선택
```

- **StratifiedKFold** → Cross Validation을 위한 **데이터 분할 방법**
- **GridSearchCV** → 모든 지정 후보를 **Cross Validation으로 평가하여 하이퍼파라미터 탐색**
- **RandomizedSearchCV** → 무작위로 선택한 후보를 **Cross Validation으로 평가하여 하이퍼파라미터 탐색**



# 5장 2강: 누수 없이 학습하고 재사용하는 End-to-End Pipeline

```
from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline([ ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ])
```
- Pipeline은 클래스
- `steps` 매개변수에 **`(단계 이름, 변환기/모델 객체)` 형태의 튜플을 담은 리스트**를 전달
	```
	sklearn
  └─ pipeline
       └─ Pipeline 클래스
    
    # `steps`에 `(이름, 객체)` 튜플의 리스트를 전달
    Pipeline(
    steps=[
        ("imputer", SimpleImputer()),
        ("onehot", OneHotEncoder())
    ]
)
	```


### Q. 왜  sklearn 코드에서 딕셔너리가 아닌 `튜플`을 사용하는지?
#### 튜플과 딕셔너리 차이점

| 구분      | Tuple              | Dictionary         |
| ------- | ------------------ | ------------------ |
| 형태      | `("imputer", obj)` | `{"imputer": obj}` |
| 핵심 개념   | **값들의 묶음 / 한 레코드** | **Key → Value 매핑** |
| 값의 의미   | 위치로 구분             | Key로 구분            |
| 접근      | `x[0]`, `x[1]`     | `x["imputer"]`     |
| 중복 이름   | 가능                 | Key 중복 불가          |
| 변경      | 불변(immutable)      | 변경 가능(mutable)     |
| 잘 맞는 상황 | **정해진 구조의 한 항목**   | **이름으로 값을 검색·관리**  |

#### Pipeline에서는 왜 Tuple이 자연스러운가?

- **Tuple**: 정해진 구조의 여러 값을 **하나의 레코드(묶음)**로 표현할 때 적합
- **Dictionary**: `Key → Value` 형태로 **이름을 통해 값을 조회·관리**할 때 적합
- Pipeline의 한 Step은 항상 **`이름(name)` + `객체(estimator)`**라는 고정된 두 값으로 구성
  - `("imputer", SimpleImputer())`
  - `("scale", StandardScaler())`
- 따라서 하나의 Step을 `(name, estimator)` **Tuple**로 묶는 구조가 간결함
- Tuple은 **언패킹(Unpacking)**이 쉬움
  - `for name, estimator in steps:`
- 여러 Step의 **실행 순서**는 List로 표현
  - `[("imputer", ...), ("scale", ...), ("model", ...)]`
- 이름으로 특정 Step을 조회해야 할 때는 별도로 Mapping 형태의 `named_steps` 제공
  - `pipeline.named_steps["imputer"]`

> **정리: Tuple = 하나의 Step을 구성하는 고정된 값의 묶음 / List = Step들의 실행 순서 / Mapping = 이름을 통한 Step 조회**


 ### Q. Logistic Regression 규제 강도 역수 C

### 

### 규제 강도 역수 C

#### 규제
- **목적**: 과적합(Overfitting) 방지를 위해 모델의 복잡도 제한
- **방법**: 손실함수에 규제항(Penalty)을 추가하여 큰 가중치 억제

| 규제 | 특징 |
|---|---|
| **L1 (Lasso)** | 일부 가중치를 0으로 만들어 Feature 선택 효과 |
| **L2 (Ridge)** | 가중치를 전반적으로 작게 억제 |
### Logistic Regression
- `penalty` : 규제 종류 (`l1`, `l2`)
- `C` : **규제 강도의 역수**
  - `C ↓` → 규제 ↑
  - `C ↑` → 규제 ↓
- 기본값: `C=1.0`, `penalty="l2"`
- C 탐색: `0.001 → 0.01 → 0.1 → 1 → 10 → 100`
  - 넓은 범위를 효율적으로 탐색하기 위한 **로그 스케일(Log Scale)**
### Logistic Regression의 C와 규제

- 일반적인 규제 표현: `Loss + λ × Penalty`
  - `λ ↑` → 규제 강함
  - `λ ↓` → 규제 약함

- Logistic Regression에서는 규제 조절값으로 **C** 사용
- C는 규제 강도와 **반대 방향(역수 관계)**

`C ∝ 1 / λ`

- `C ↓` → 규제 ↑
- `C ↑` → 규제 ↓

> **C를 역수로 써야 하는 수학적 필연성이 있는 것은 아니며, Logistic Regression에서 규제 강도와 반대 방향으로 정의한 하이퍼파라미터이다.**

# 데일리 퀘스트 오답

## Q. Optuna가 기본으로 사용하는 TPE(Tree-structured Parzen Estimator) 방식

**정의**: 이전에 시도한 결과를 보고, 성능이 좋을 가능성이 높은 하이퍼파라미터 영역을 다음에 더 탐색하는 방법

```
초기 Trial
C=0.01 → AP 0.91
C=0.1  → AP 0.98  ← 좋음
C=1    → AP 0.93
       ↓
TPE가 이전 결과 분석
       ↓
0.1 주변이 유망하다고 판단
       ↓
C=0.07, 0.15 ... 등 유망한 값 탐색
```

|방법|다음 값을 고르는 방식|
|---|---|
|Grid Search|미리 정한 값 전부 탐색|
|Random Search|무작위 선택|
|**TPE**|**이전 Trial 결과를 이용해 유망한 영역 선택**|
- 무작정 모든 값을 찾아보지 않고 이전 실험에서 학습하면서 다음 하이퍼파라미터를 선택
- `Tree-structured`가 들어가지만 **Decision Tree나 Random Forest의 Tree를 만드는 알고리즘은 아님

## 오늘 공부 요약
![](../../첨부파일/260818_하이퍼파라미터%20종류와%20개념.png)



