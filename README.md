# 진동·전류 시계열 기반 프레스 유압펌프 이상 탐지

## 1. 프로젝트 개요

본 프로젝트는 프레스 설비의 유압펌프 모터에서 수집된 진동 및 전류 시계열 데이터를 활용하여 정상/이상 상태를 분류하고, 오경보(False Positive)와 미탐지(False Negative)의 발생 조건을 분석하는 것을 목적으로 한다.

단순 분류 성능뿐만 아니라 모델의 오류 특성을 분석하고, 최종적으로 현장 점검 우선순위에 활용할 수 있는 RED / YELLOW / GREEN 단계별 경보 체계를 구성하였다.

---

## 2. 프로젝트 구조

```text
project/
├─ main.py
├─ run_analysis.py
├─ README.md
├─ requirements.txt
│
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ preprocessing.py
│  ├─ features.py
│  ├─ modeling.py
│  └─ alarm.py
│
├─ analysis/
│  ├─ __init__.py
│  ├─ dda.py
│  ├─ eda.py
│  ├─ gap_sensitivity.py
│  ├─ influence_analysis.py
│  ├─ error_analysis.py
│  └─ abnormal3_analysis.py
│
├─ data/
├─ results/
└─ figures/
```

- `src/`: 최종 결과를 생성하는 핵심 파이프라인 코드
- `analysis/`: DDA, EDA, Gap 민감도, 영향요인 및 오류분석 등 보고서용 보조 분석 코드

---

## 3. 데이터

본 프로젝트는 **KAMP(인공지능 제조 플랫폼)에 등재된 제조AI데이터셋**을 활용하였다.

분석에 사용한 원본 파일:

```text
press_data_normal.csv
outlier_data.csv
```

주요 변수:

| 변수 | 설명 |
|---|---|
| `TimeStamp` | 센서 측정 시각 |
| `AI0_Vibration` | 진동 센서 신호 |
| `AI1_Vibration` | 진동 센서 신호 |
| `AI2_Current` | 전류 센서 신호 |
| `Equipment_state` | 정상(0) / 이상(1) 상태 |

AI0와 AI1 센서의 정확한 물리적 설치 위치에 대한 별도의 설명 자료가 없어 임의로 상부/하부 등의 의미를 부여하지 않고 원래 변수명을 그대로 사용하였다.

### 원본 데이터 공개 안내

원본 CSV는 공모전 및 KAMP에서 제공된 데이터이며, **원본 데이터의 재배포 권한이 명확히 확인되지 않아 본 GitHub 저장소에는 포함하지 않는다.**

실행 시 아래 파일을 로컬 `data/` 폴더에 직접 배치해야 한다.

```text
data/
├─ press_data_normal.csv
└─ outlier_data.csv
```

`.gitignore`:

```gitignore
data/*.csv
```

코드, 분석 결과 및 시각화 자료는 저장소에 포함하되 원본 데이터는 KAMP 제공 경로를 통해 별도로 확보한다.

---

## 4. 실행 환경

권장 및 검증 환경:

```text
Python 3.12.x
```

주요 라이브러리:

```text
numpy==2.3.5
pandas==2.2.3
scikit-learn==1.8.0
matplotlib==3.10.8
scipy==1.18.0
```

설치:

```bash
python -m pip install -r requirements.txt
```

---

## 5. 실행 방법

원본 CSV를 `data/` 폴더에 배치한 후 프로젝트 루트에서 실행한다.

```bash
python main.py
```

`main.py` 실행 흐름:

```text
원본 데이터 로드
→ 데이터 전처리
→ Segment 생성
→ Feature Set A/B/C/D 생성
→ 5-Fold 모델 학습 및 검증 추론
→ OOF Prediction 생성
→ Two-stage Alarm 생성
```

추가적인 분석 결과와 보고서용 시각화를 재현하려면:

```bash
python run_analysis.py
```

---

## 6. 전처리 및 Segment 구성

- 불필요한 `Unnamed` 계열 컬럼 제거
- 정상 데이터 중복 1건 제거
- `TimeStamp` datetime 변환 및 시간순 정렬
- 주요 센서 변수 결측 여부 확인
- 센서 극단값은 이상 신호일 가능성이 있어 일괄 제거하지 않음

원시 시계열의 주요 측정 간격은 약 **0.1초**였으나 일부 구간에 큰 시간 공백이 존재하였다.

긴 시간 공백을 보간하면 실제 측정되지 않은 파형을 인위적으로 생성할 수 있으므로 보간 대신 Segment를 분리하였다.

```text
시간 차이 ≤ 0.5초 → 동일 Segment
시간 차이 > 0.5초 → 새로운 Segment
```

최종 Segment:

| 상태 | Segment 수 |
|---|---:|
| Normal | 599 |
| Abnormal | 21 |
| Total | 620 |

Gap threshold 0.2 / 0.3 / 0.5 / 1.0초를 비교한 결과 Segment 수와 분할 경계가 모두 동일하였다. 따라서 0.5초를 최적값이 아닌 대표 기준으로 사용하였다.

---

## 7. Feature Engineering

각 Segment를 하나의 학습 샘플로 구성하였다.

### Set A — Basic Statistical Features
각 센서별:
- mean
- std
- RMS
- max absolute
- range

총 **15개 Feature**

### Set B — Distribution / Impulse Features
Set A에 다음 Feature 추가:
- median
- skew
- kurtosis
- crest factor

총 **27개 Feature**

### Set C — Sensor Relationship Features
Set A에 다음 6개 관계 Feature 추가:
- `Vibration_total_rms`
- `Current_rms_to_Vibration_rms`
- `Current_mean_to_Vibration_rms`
- `Current_max_to_Vibration_rms`
- `AI0_to_AI1_rms_ratio`
- `Current_std_to_Vibration_std`

총 **21개 Feature**

### Set D — Temporal Features
Set C에 다음 7개 시간 변화 Feature 추가:
- `AI0_Vibration_slope`
- `AI0_Vibration_diff_rms`
- `AI1_Vibration_slope`
- `AI1_Vibration_diff_rms`
- `AI2_Current_slope`
- `AI2_Current_diff_rms`
- `AI0_AI1_corr`

총 **28개 Feature**

모델 입력 제외 변수:

```text
sample_id
segment
length
label
```

---

## 8. 모델링 및 검증

### Logistic Regression

```text
SimpleImputer(median)
→ StandardScaler
→ LogisticRegression
```

설정:
- `class_weight="balanced"`
- `max_iter=3000`
- `random_state=42`

### Random Forest

```text
SimpleImputer(median)
→ RandomForestClassifier
```

설정:
- `n_estimators=300`
- `class_weight="balanced"`
- `random_state=42`

검증 방식:

```text
StratifiedKFold
n_splits = 5
shuffle = True
random_state = 42
```

모든 Feature Set과 모델은 동일한 Fold 조건에서 비교하였다.

주요 평가 지표:
- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- FP / FN

---

## 9. 모델 성능

| Feature Set | Model | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| A | Logistic Regression | 0.8182 | 0.8571 | 0.8372 |
| A | Random Forest | 1.0000 | 0.8571 | 0.9231 |
| B | Logistic Regression | 0.8571 | 0.8571 | 0.8571 |
| B | Random Forest | 1.0000 | 0.8571 | 0.9231 |
| C | Logistic Regression | 0.9048 | 0.9048 | 0.9048 |
| C | Random Forest | 1.0000 | 0.8571 | 0.9231 |
| **D** | **Logistic Regression** | **1.0000** | **0.9524** | **0.9756** |
| D | Random Forest | 1.0000 | 0.8571 | 0.9231 |

최종 모델:

```text
Feature Set D + Logistic Regression
```

OOF Confusion Matrix:

```text
TN = 599
FP = 0
FN = 1
TP = 20
```

최종 주요 성능:

```text
Precision = 1.0000
Recall    = 0.9524
F1-score  = 0.9756
PR-AUC    = 0.9940
ROC-AUC   = 0.9998
```

---

## 10. 오류 분석

최종 Set D Logistic Regression은 정상 599개에 대해 FP가 발생하지 않았으며, 이상 21개 중 20개를 탐지하였다.

최종 미탐 샘플:

```text
abnormal_3
```

해당 Segment는 진동 Feature가 정상 영역과 상당 부분 겹치는 반면 전류 변화량은 매우 크게 나타나는 혼합형 패턴을 보였다.

시간 Feature를 추가하면서 이상 확률은 증가했으나 최종 threshold 0.5를 넘지는 못하였다.

따라서 본 분석에서는 단순 모델 성능뿐 아니라 **모델이 어떤 조건에서 실패하는지까지 분석**하였다.

---

## 11. Two-stage Alarm

현장 활용을 위해 Random Forest와 Set D Logistic Regression의 예측 결과를 결합하여 3단계 경보 체계를 구성하였다.

- **RED**: Random Forest = Abnormal
- **YELLOW**: Random Forest = Normal & Set D Logistic Regression = Abnormal
- **GREEN**: 두 모델 모두 Normal

결과:

| Alarm | Normal | Abnormal |
|---|---:|---:|
| GREEN | 599 | 1 |
| YELLOW | 0 | 2 |
| RED | 0 | 18 |

Two-stage Alarm의 목적은 Set D Logistic Regression보다 이진 분류 성능을 높이는 것이 아니라, **동일한 예측 결과를 현장 점검 우선순위로 구분하는 것**이다.

---

## 12. 결과 파일

`main.py` 실행 후 주요 결과:

```text
results/
├─ oof_predictions.csv
└─ two_stage_alarm/
   └─ two_stage_alarm_results.csv
```

`oof_predictions.csv`는 독립 Test Set 결과가 아니라 **5-Fold Cross Validation을 통해 생성된 Out-of-Fold Prediction 결과**이다.

별도의 독립 Test CSV가 제공되지 않았기 때문에 OOF 결과를 독립 Test 성능으로 표현하지 않는다.

---

## 13. 한계점

1. 이상 Segment가 21개로 매우 적다.
2. 정상과 이상 데이터가 서로 다른 수집 세션에서 확보되었다.
3. 세션 또는 운전 조건 차이가 모델에 영향을 주었을 가능성이 있다.
4. Set C와 Set D Feature는 기존 오류를 확인한 이후 설계한 탐색적 Feature이다.
5. 현재 데이터에서의 개선 효과가 독립 데이터에서도 동일하게 재현된다고 단정할 수 없다.
6. 정확한 고장 발생 시점이 제공되지 않아 실제 Lead-time 성능은 평가하지 않았다.

---

## 14. 재현성

핵심 파이프라인:

```bash
python main.py
```

보고서용 추가 분석:

```bash
python run_analysis.py
```

```text
전처리
→ Segment 생성
→ Feature Engineering
→ 5-Fold 모델 학습 및 검증
→ OOF Prediction
→ 오류 분석
→ 최종 Alarm 생성
```

---

## 15. 주의사항

본 저장소는 분석 과정과 코드를 공유하기 위한 프로젝트이다.

KAMP에서 제공된 원본 제조AI데이터셋은 재배포 권한이 명확하지 않아 GitHub 저장소에 포함하지 않는다.

원본 데이터가 필요한 경우 KAMP의 공식 데이터 제공 경로를 이용해야 한다.
