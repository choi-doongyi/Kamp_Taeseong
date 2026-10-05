# 기존 코드 → 제출용 모듈 매핑

기존 분석 로직을 없앤 것이 아니라 제출용으로 역할별 통합했습니다.

| 기존 파일 | 제출용 모듈 |
|---|---|
| `01_dda.py` | `analysis/dda.py` |
| `02_preprocessing.py` | `src/preprocessing.py` |
| `03_eda.py` | `analysis/eda.py` |
| `feature_engineering.py` | `src/features.py` |
| `feature_set_c.py` | `src/features.py`의 `make_feature_set_c()` |
| `feature_set_d.py` | `src/features.py`의 `make_feature_set_d()` |
| `modeling.py` | `src/modeling.py` |
| `error_analysis.py` | `analysis/error_analysis.py`, `analysis/influence_analysis.py` |
| `error_analysis_setc_logistic.py` | `analysis/error_analysis.py`의 Set C ↔ Set D 비교 |
| `error_analysis_setd_logistic.py` | `analysis/error_analysis.py` |
| `two_stage_alarm.py` | `src/alarm.py` |
| `abnormal3_analysis.py` | `analysis/abnormal3_analysis.py` |
| `gap_sensitivity.py` | `analysis/gap_sensitivity.py` |

## 핵심 변경점

1. `DDA → clean CSV → preprocessing`의 중간 의존성을 제거했습니다. `src/preprocessing.py`가 원본 CSV에서 바로 시작합니다.
2. Set A/B/C/D 생성 로직을 `src/features.py` 하나로 통합했습니다.
3. A/B/C/D × Logistic/RF는 `src/modeling.py`가 동일한 Stratified 5-Fold로 자동 비교합니다.
4. 최종 OOF 예측을 `results/oof_predictions.csv`로 자동 복사합니다.
5. 최종 경보는 `Set A RF + Set D Logistic` 조합으로 고정했습니다.
6. `main.py` 하나로 전처리→Feature→모델링→경보 결과생성까지 실행됩니다.
7. 보고서용 DDA/EDA/오류분석은 핵심 실행과 분리하여 `run_analysis.py`로 재현합니다.

## 주의

현재 전달 패키지에는 원본 CSV 자체가 포함되어 있지 않습니다. 제출 전 사용자가 보유한 원본 학습 데이터 2개를 `data/` 폴더에 넣어야 합니다.
