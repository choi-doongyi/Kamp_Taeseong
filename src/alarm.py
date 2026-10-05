from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .config import ALARM_DIR, MODELING_DIR, RESULT_DIR, ensure_directories

RF_PATH = MODELING_DIR / "SetA_RandomForest_predictions.csv"
SETD_LR_PATH = MODELING_DIR / "SetD_LogisticRegression_predictions.csv"


def load_predictions():
    rf = pd.read_csv(RF_PATH)
    setd_lr = pd.read_csv(SETD_LR_PATH)

    if not rf["sample_id"].equals(setd_lr["sample_id"]):
        raise ValueError("RF와 Set D Logistic의 sample_id 순서가 다릅니다.")
    if not rf["actual"].equals(setd_lr["actual"]):
        raise ValueError("RF와 Set D Logistic의 실제 label이 다릅니다.")
    return rf, setd_lr


def make_alarm_levels(rf: pd.DataFrame, setd_lr: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "sample_id": rf["sample_id"],
            "segment": rf["segment"],
            "length": rf["length"],
            "actual": rf["actual"],
            "rf_probability": rf["probability"],
            "rf_prediction": rf["prediction"],
            "setd_probability": setd_lr["probability"],
            "setd_prediction": setd_lr["prediction"],
        }
    )

    red = result["rf_prediction"] == 1
    yellow = (result["rf_prediction"] == 0) & (result["setd_prediction"] == 1)
    result["alarm_level"] = np.select([red, yellow], ["RED", "YELLOW"], default="GREEN")
    result["final_prediction"] = result["alarm_level"].isin(["RED", "YELLOW"]).astype(int)
    return result


def evaluate_alarm(result: pd.DataFrame) -> dict[str, float]:
    y_true = result["actual"]
    y_pred = result["final_prediction"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "fnr": float(fn / (fn + tp) if (fn + tp) else 0.0),
    }
    return metrics


def run_two_stage_alarm():
    ensure_directories()
    print("\n" + "=" * 60)
    print("TWO-STAGE ALARM START")
    print("=" * 60)

    rf, setd_lr = load_predictions()
    result = make_alarm_levels(rf, setd_lr)
    metrics = evaluate_alarm(result)

    result.to_csv(ALARM_DIR / "two_stage_alarm_results.csv", index=False)
    pd.DataFrame([metrics]).to_csv(ALARM_DIR / "two_stage_alarm_metrics.csv", index=False)

    alarm_table = pd.crosstab(result["alarm_level"], result["actual"]).reindex(
        ["GREEN", "YELLOW", "RED"], fill_value=0
    )
    alarm_table = alarm_table.rename(columns={0: "Normal", 1: "Abnormal"})
    alarm_table.to_csv(ALARM_DIR / "alarm_level_distribution.csv")

    for level in ["RED", "YELLOW"]:
        result[result["alarm_level"] == level].to_csv(
            ALARM_DIR / f"{level.lower()}_samples.csv", index=False
        )
    result[(result["alarm_level"] == "GREEN") & (result["actual"] == 1)].to_csv(
        ALARM_DIR / "green_abnormal_samples.csv", index=False
    )

    # 제출용 최종 결과 요약
    result[[
        "sample_id", "actual", "final_prediction", "alarm_level",
        "rf_probability", "setd_probability"
    ]].to_csv(RESULT_DIR / "prediction_results.csv", index=False)

    print("\nFINAL TWO-STAGE PERFORMANCE")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("\nALARM LEVEL DISTRIBUTION")
    print(alarm_table.to_string())
    print("TWO-STAGE ALARM COMPLETE")
    return result, metrics


if __name__ == "__main__":
    run_two_stage_alarm()
