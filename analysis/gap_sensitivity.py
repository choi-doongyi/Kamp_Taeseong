from __future__ import annotations

import pandas as pd

from src.config import ABNORMAL_RAW_PATH, GAP_DIR, NORMAL_RAW_PATH, ensure_directories
from src.preprocessing import basic_cleaning

GAP_THRESHOLDS = [0.2, 0.3, 0.5, 1.0]
BASELINE_GAP = 0.5


def make_segments(df: pd.DataFrame, gap_seconds: float) -> pd.DataFrame:
    temp = df.sort_values("TimeStamp").reset_index(drop=True).copy()
    temp["time_diff"] = temp["TimeStamp"].diff().dt.total_seconds()
    temp["new_segment"] = temp["time_diff"] > gap_seconds
    temp["segment"] = temp["new_segment"].cumsum().astype(int)
    return temp


def _break_timestamps(segmented: pd.DataFrame) -> set:
    return set(segmented.loc[segmented["new_segment"], "TimeStamp"])


def analyze_dataset(df: pd.DataFrame, dataset_name: str):
    segmented_dict = {}
    stats_rows = []

    for gap in GAP_THRESHOLDS:
        segmented = make_segments(df, gap)
        segmented_dict[gap] = segmented
        lengths = segmented.groupby("segment").size()
        stats_rows.append(
            {
                "dataset": dataset_name,
                "gap_threshold": gap,
                "segment_count": int(lengths.size),
                "break_count": int(segmented["new_segment"].sum()),
                "short_segment_lt10": int((lengths < 10).sum()),
                "min_length": int(lengths.min()),
                "q1_length": float(lengths.quantile(0.25)),
                "median_length": float(lengths.median()),
                "mean_length": float(lengths.mean()),
                "q3_length": float(lengths.quantile(0.75)),
                "max_length": int(lengths.max()),
            }
        )

    baseline = _break_timestamps(segmented_dict[BASELINE_GAP])
    boundary_rows = []
    for gap in GAP_THRESHOLDS:
        current = _break_timestamps(segmented_dict[gap])
        boundary_rows.append(
            {
                "dataset": dataset_name,
                "gap_threshold": gap,
                "baseline_gap": BASELINE_GAP,
                "same_break_count": len(current & baseline),
                "added_break_count": len(current - baseline),
                "removed_break_count": len(baseline - current),
                "exact_same_boundaries": current == baseline,
            }
        )

    return pd.DataFrame(stats_rows), pd.DataFrame(boundary_rows)


def run_gap_sensitivity():
    ensure_directories()
    normal_raw = pd.read_csv(NORMAL_RAW_PATH)
    abnormal_raw = pd.read_csv(ABNORMAL_RAW_PATH)
    normal, _ = basic_cleaning(normal_raw)
    abnormal, _ = basic_cleaning(abnormal_raw)

    n_stats, n_boundary = analyze_dataset(normal, "normal")
    a_stats, a_boundary = analyze_dataset(abnormal, "abnormal")
    stats = pd.concat([n_stats, a_stats], ignore_index=True)
    boundary = pd.concat([n_boundary, a_boundary], ignore_index=True)

    stats.to_csv(GAP_DIR / "gap_sensitivity_stats.csv", index=False)
    boundary.to_csv(GAP_DIR / "gap_boundary_comparison.csv", index=False)
    print("\nSEGMENT COUNT COMPARISON")
    print(stats.round(3).to_string(index=False))
    print("\nBOUNDARY COMPARISON VS 0.5 SEC")
    print(boundary.to_string(index=False))
    return stats, boundary


if __name__ == "__main__":
    run_gap_sensitivity()
