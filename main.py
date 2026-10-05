"""One-command reproducible pipeline.

Run from the project root:
    python main.py
"""

from src.alarm import run_two_stage_alarm
from src.features import run_feature_engineering
from src.modeling import run_modeling
from src.preprocessing import run_preprocessing


def main():
    print("\n" + "#" * 70)
    print("HYDRAULIC PUMP ANOMALY DETECTION PIPELINE")
    print("#" * 70)

    run_preprocessing()
    run_feature_engineering()
    comparison = run_modeling()
    alarm_result, alarm_metrics = run_two_stage_alarm()

    print("\n" + "#" * 70)
    print("PIPELINE COMPLETE")
    print("#" * 70)
    print("최종 모델: Set D + Logistic Regression")
    print("현장 경보 보조 모델: Set A + Random Forest")
    print("주요 결과 파일: results/oof_predictions.csv")
    print("경보 결과 파일: results/two_stage_alarm/two_stage_alarm_results.csv")
    return comparison, alarm_result, alarm_metrics


if __name__ == "__main__":
    main()
