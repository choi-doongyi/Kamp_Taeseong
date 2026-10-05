from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ABNORMAL_RAW_PATH, ANALYSIS_DIR, NORMAL_RAW_PATH, SENSOR_COLS, ensure_directories
from src.preprocessing import basic_cleaning


def _summarize(df: pd.DataFrame, name: str) -> dict:
    clean, removed_duplicates = basic_cleaning(df)
    diffs = clean["TimeStamp"].diff().dt.total_seconds().dropna()
    rounded = diffs.round(3)
    mode_interval = float(rounded.mode().iloc[0]) if not rounded.empty else np.nan
    non_main = diffs[~np.isclose(diffs, 0.1, atol=0.001)]

    row = {
        "dataset": name,
        "rows_raw": len(df),
        "rows_clean": len(clean),
        "removed_duplicates": removed_duplicates,
        "missing_total": int(clean.isna().sum().sum()),
        "timestamp_min": clean["TimeStamp"].min(),
        "timestamp_max": clean["TimeStamp"].max(),
        "main_interval_sec": mode_interval,
        "non_0.1_interval_count": int(len(non_main)),
        "other_interval_min_sec": float(non_main.min()) if len(non_main) else np.nan,
        "other_interval_max_sec": float(non_main.max()) if len(non_main) else np.nan,
    }

    print(f"\n[{name}] shape={clean.shape}, duplicates removed={removed_duplicates}")
    print("Label distribution:")
    print(clean["Equipment_state"].value_counts(dropna=False).to_string())
    print("Sensor describe:")
    print(clean[SENSOR_COLS].describe().T.round(6).to_string())
    print("Top timestamp intervals:")
    print(rounded.value_counts().head(20).to_string())
    return row


def run_dda():
    ensure_directories()
    normal = pd.read_csv(NORMAL_RAW_PATH)
    abnormal = pd.read_csv(ABNORMAL_RAW_PATH)
    rows = [_summarize(normal, "normal"), _summarize(abnormal, "abnormal")]
    summary = pd.DataFrame(rows)
    summary.to_csv(ANALYSIS_DIR / "dda_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    run_dda()
