from __future__ import annotations

import pandas as pd

from src.config import ANALYSIS_DIR, MODELING_DIR, ensure_directories


def _summarize(path, value_col: str, output_name: str, use_abs: bool = False):
    df = pd.read_csv(path)
    if use_abs:
        df["abs_value"] = df[value_col].abs()
        grouped = df.groupby("feature").agg(
            mean_value=(value_col, "mean"),
            mean_abs_value=("abs_value", "mean"),
            std_value=(value_col, "std"),
        ).sort_values("mean_abs_value", ascending=False)
    else:
        grouped = df.groupby("feature").agg(
            mean_value=(value_col, "mean"),
            std_value=(value_col, "std"),
        ).sort_values("mean_value", ascending=False)
    grouped.to_csv(ANALYSIS_DIR / output_name)
    return grouped


def run_influence_analysis():
    ensure_directories()
    rf_path = MODELING_DIR / "SetA_RandomForest_fold_interpretation.csv"
    lr_path = MODELING_DIR / "SetD_LogisticRegression_fold_interpretation.csv"

    rf_summary = _summarize(
        rf_path, "importance", "seta_rf_feature_importance_summary.csv", use_abs=False
    )
    lr_summary = _summarize(
        lr_path, "coefficient", "setd_logistic_coefficient_summary.csv", use_abs=True
    )

    print("\nSet A Random Forest feature importance")
    print(rf_summary.head(15).round(6).to_string())
    print("\nSet D Logistic coefficient magnitude")
    print(lr_summary.head(15).round(6).to_string())
    return rf_summary, lr_summary


if __name__ == "__main__":
    run_influence_analysis()
