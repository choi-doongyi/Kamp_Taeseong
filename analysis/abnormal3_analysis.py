from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    ABNORMAL_PREPROCESSED_PATH,
    ANALYSIS_DIR,
    MODELING_DIR,
    NORMAL_PREPROCESSED_PATH,
    SET_D_PATH,
    ensure_directories,
)
from src.features import TEMPORAL_FEATURES

TARGET_SAMPLE_ID = "abnormal_3"
CORE_FEATURES = [
    "AI0_Vibration_rms",
    "AI1_Vibration_rms",
    "AI2_Current_mean",
    "AI2_Current_rms",
    "Vibration_total_rms",
    "Current_mean_to_Vibration_rms",
    "Current_rms_to_Vibration_rms",
    "AI0_to_AI1_rms_ratio",
]


def run_abnormal3_analysis():
    ensure_directories()
    features = pd.read_csv(SET_D_PATH)
    lr = pd.read_csv(MODELING_DIR / "SetD_LogisticRegression_predictions.csv")
    rf = pd.read_csv(MODELING_DIR / "SetA_RandomForest_predictions.csv")

    target = features[features["sample_id"] == TARGET_SAMPLE_ID]
    if target.empty:
        raise ValueError(f"{TARGET_SAMPLE_ID}을 찾을 수 없습니다.")
    target = target.iloc[0]

    normal = features[features["label"] == 0]
    rows = []
    for feature in [*CORE_FEATURES, *TEMPORAL_FEATURES]:
        normal_values = normal[feature].dropna()
        value = target[feature]
        rows.append(
            {
                "sample_id": TARGET_SAMPLE_ID,
                "feature": feature,
                "value": value,
                "normal_median": normal_values.median(),
                "normal_percentile": float((normal_values <= value).mean() * 100),
            }
        )
    percentile_df = pd.DataFrame(rows)
    percentile_df.to_csv(ANALYSIS_DIR / "abnormal3_normal_percentile.csv", index=False)

    prediction = (
        rf[rf["sample_id"] == TARGET_SAMPLE_ID][["sample_id", "probability"]]
        .rename(columns={"probability": "rf_probability"})
        .merge(
            lr[lr["sample_id"] == TARGET_SAMPLE_ID][["sample_id", "probability"]].rename(
                columns={"probability": "setd_probability"}
            ),
            on="sample_id",
        )
    )
    prediction.to_csv(ANALYSIS_DIR / "abnormal3_model_probabilities.csv", index=False)

    # Raw segment summary for traceability.
    abnormal = pd.read_csv(ABNORMAL_PREPROCESSED_PATH, parse_dates=["TimeStamp"])
    segment_id = int(TARGET_SAMPLE_ID.split("_")[-1])
    raw_segment = abnormal[abnormal["segment"] == segment_id].copy()
    raw_segment.to_csv(ANALYSIS_DIR / "abnormal3_raw_segment.csv", index=False)

    print(percentile_df.round(6).to_string(index=False))
    print("\nModel probabilities")
    print(prediction.round(6).to_string(index=False))
    return percentile_df, prediction


if __name__ == "__main__":
    run_abnormal3_analysis()
