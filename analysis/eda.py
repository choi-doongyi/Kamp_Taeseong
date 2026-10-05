from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    ABNORMAL_PREPROCESSED_PATH,
    FIGURE_DIR,
    NORMAL_PREPROCESSED_PATH,
    SENSOR_COLS,
    ensure_directories,
)


def load_data():
    normal = pd.read_csv(NORMAL_PREPROCESSED_PATH, parse_dates=["TimeStamp"])
    abnormal = pd.read_csv(ABNORMAL_PREPROCESSED_PATH, parse_dates=["TimeStamp"])
    return normal, abnormal


def _save_distribution_plots(normal: pd.DataFrame, abnormal: pd.DataFrame) -> None:
    for col in SENSOR_COLS:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(normal[col].dropna(), bins=50, density=True, alpha=0.5, label="Normal")
        ax.hist(abnormal[col].dropna(), bins=50, density=True, alpha=0.5, label="Abnormal")
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"eda_{col}_hist.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(
            [normal[col].dropna(), abnormal[col].dropna()],
            tick_labels=["Normal", "Abnormal"],
        )
        ax.set_title(f"{col} Boxplot")
        ax.set_ylabel(col)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"eda_{col}_boxplot.png", dpi=150)
        plt.close(fig)


def _longest_segment(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    lengths = df.groupby("segment").size()
    segment_id = int(lengths.idxmax())
    return segment_id, df[df["segment"] == segment_id].copy()


def _save_representative_timeseries(normal: pd.DataFrame, abnormal: pd.DataFrame) -> None:
    n_id, n_group = _longest_segment(normal)
    a_id, a_group = _longest_segment(abnormal)

    for col in SENSOR_COLS:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(len(n_group)), n_group[col].to_numpy(), label=f"Normal segment {n_id}")
        ax.plot(np.arange(len(a_group)), a_group[col].to_numpy(), label=f"Abnormal segment {a_id}")
        ax.set_title(f"Representative Segment - {col}")
        ax.set_xlabel("Sample order within segment")
        ax.set_ylabel(col)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"eda_{col}_representative_segment.png", dpi=150)
        plt.close(fig)


def _basic_segment_summary(df: pd.DataFrame, label_name: str) -> pd.DataFrame:
    rows = []
    for segment, group in df.groupby("segment"):
        row = {"dataset": label_name, "segment": int(segment), "length": len(group)}
        for col in SENSOR_COLS:
            values = group[col].astype(float).to_numpy()
            row[f"{col}_mean"] = float(np.mean(values))
            row[f"{col}_std"] = float(np.std(values, ddof=0))
            row[f"{col}_rms"] = float(np.sqrt(np.mean(values ** 2)))
            row[f"{col}_max_abs"] = float(np.max(np.abs(values)))
            row[f"{col}_range"] = float(np.max(values) - np.min(values))
        rows.append(row)
    return pd.DataFrame(rows)


def run_eda():
    ensure_directories()
    normal, abnormal = load_data()
    _save_distribution_plots(normal, abnormal)
    _save_representative_timeseries(normal, abnormal)

    summary = pd.concat(
        [_basic_segment_summary(normal, "normal"), _basic_segment_summary(abnormal, "abnormal")],
        ignore_index=True,
    )
    summary.to_csv(FIGURE_DIR / "eda_segment_basic_features.csv", index=False)

    mean_table = summary.groupby("dataset").mean(numeric_only=True).T
    mean_table.to_csv(FIGURE_DIR / "eda_normal_abnormal_feature_means.csv")
    print(mean_table.round(6).to_string())
    print(f"EDA figures saved to: {FIGURE_DIR}")
    return summary


if __name__ == "__main__":
    run_eda()
