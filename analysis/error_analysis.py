from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ANALYSIS_DIR, MODELING_DIR, SET_D_PATH, ensure_directories
from src.features import RELATION_FEATURES, TEMPORAL_FEATURES

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


def _load():
    set_d = pd.read_csv(SET_D_PATH)
    setd_lr = pd.read_csv(MODELING_DIR / "SetD_LogisticRegression_predictions.csv")
    setc_lr = pd.read_csv(MODELING_DIR / "SetC_LogisticRegression_predictions.csv")
    seta_rf = pd.read_csv(MODELING_DIR / "SetA_RandomForest_predictions.csv")
    return set_d, setd_lr, setc_lr, seta_rf


def _prediction_compare(left, right, left_name: str, right_name: str) -> pd.DataFrame:
    l = left[["sample_id", "actual", "probability", "prediction", "result_type"]].rename(
        columns={
            "probability": f"{left_name}_probability",
            "prediction": f"{left_name}_prediction",
            "result_type": f"{left_name}_result",
        }
    )
    r = right[["sample_id", "probability", "prediction", "result_type"]].rename(
        columns={
            "probability": f"{right_name}_probability",
            "prediction": f"{right_name}_prediction",
            "result_type": f"{right_name}_result",
        }
    )
    return l.merge(r, on="sample_id", how="inner", validate="one_to_one")


def run_error_analysis():
    ensure_directories()
    set_d, setd_lr, setc_lr, seta_rf = _load()

    merged = setd_lr.merge(
        set_d,
        on="sample_id",
        how="left",
        suffixes=("_pred", "_feature"),
        validate="one_to_one",
    )

    fp = merged[merged["result_type"] == "FP"].copy()
    fn = merged[merged["result_type"] == "FN"].copy()
    fp.to_csv(ANALYSIS_DIR / "setd_logistic_fp.csv", index=False)
    fn.to_csv(ANALYSIS_DIR / "setd_logistic_fn.csv", index=False)

    setc_vs_setd = _prediction_compare(setc_lr, setd_lr, "setc", "setd")
    setc_vs_setd.to_csv(ANALYSIS_DIR / "setc_vs_setd_all.csv", index=False)
    setc_vs_setd[
        (setc_vs_setd["setc_result"] == "FN") & (setc_vs_setd["setd_result"] == "TP")
    ].to_csv(ANALYSIS_DIR / "setc_fn_recovered_by_setd.csv", index=False)
    setc_vs_setd[
        (setc_vs_setd["setc_result"] == "FP") & (setc_vs_setd["setd_result"] == "TN")
    ].to_csv(ANALYSIS_DIR / "setc_fp_resolved_by_setd.csv", index=False)

    rf_vs_setd = _prediction_compare(seta_rf, setd_lr, "rf", "setd")
    rf_vs_setd.to_csv(ANALYSIS_DIR / "rf_vs_setd_all.csv", index=False)
    rf_vs_setd[
        (rf_vs_setd["rf_result"] == "FN") & (rf_vs_setd["setd_result"] == "TP")
    ].to_csv(ANALYSIS_DIR / "rf_fn_recovered_by_setd.csv", index=False)
    common_fn = rf_vs_setd[
        (rf_vs_setd["rf_result"] == "FN") & (rf_vs_setd["setd_result"] == "FN")
    ].copy()
    common_fn.to_csv(ANALYSIS_DIR / "rf_setd_common_fn.csv", index=False)

    temporal_cols = [c for c in TEMPORAL_FEATURES if c in merged.columns]
    temporal_means = merged.groupby("result_type")[temporal_cols].mean().T
    temporal_means.to_csv(ANALYSIS_DIR / "temporal_feature_group_mean.csv")

    normal = merged[merged["actual"] == 0]
    percentile_rows = []
    for _, row in fn.iterrows():
        for feature in [*CORE_FEATURES, *TEMPORAL_FEATURES]:
            if feature not in merged.columns:
                continue
            value = row[feature]
            normal_values = normal[feature].dropna()
            percentile = np.nan
            if pd.notna(value) and len(normal_values):
                percentile = float((normal_values <= value).mean() * 100)
            percentile_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "feature": feature,
                    "value": value,
                    "normal_median": normal_values.median() if len(normal_values) else np.nan,
                    "normal_percentile": percentile,
                }
            )
    percentile_df = pd.DataFrame(percentile_rows)
    percentile_df.to_csv(ANALYSIS_DIR / "final_fn_normal_percentile.csv", index=False)

    # Set A RF threshold sensitivity, using already generated OOF probabilities.
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.90]
    threshold_rows = []
    y_true = seta_rf["actual"].astype(int).to_numpy()
    prob = seta_rf["probability"].to_numpy()
    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        tn = int(((y_true == 0) & (pred == 0)).sum())
        fp_count = int(((y_true == 0) & (pred == 1)).sum())
        fn_count = int(((y_true == 1) & (pred == 0)).sum())
        tp = int(((y_true == 1) & (pred == 1)).sum())
        precision = tp / (tp + fp_count) if tp + fp_count else 0.0
        recall = tp / (tp + fn_count) if tp + fn_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        threshold_rows.append({
            "threshold": threshold, "TN": tn, "FP": fp_count, "FN": fn_count, "TP": tp,
            "precision": precision, "recall": recall, "f1": f1,
        })
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(ANALYSIS_DIR / "seta_rf_threshold_sensitivity.csv", index=False)

    print("\nSET D Logistic error distribution")
    print(setd_lr["result_type"].value_counts().to_string())
    print("\nFinal FN")
    print(fn[["sample_id", "probability", "actual", "prediction"]].to_string(index=False))
    print("\nRF / Set D common FN")
    print(common_fn.to_string(index=False))
    return {
        "setd_fp": fp,
        "setd_fn": fn,
        "setc_vs_setd": setc_vs_setd,
        "rf_vs_setd": rf_vs_setd,
        "temporal_means": temporal_means,
        "percentiles": percentile_df,
        "thresholds": threshold_df,
    }


if __name__ == "__main__":
    run_error_analysis()
