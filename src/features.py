from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ABNORMAL_PREPROCESSED_PATH,
    ALL_FEATURES_PATH,
    EPS,
    META_COLS,
    NORMAL_PREPROCESSED_PATH,
    SENSOR_COLS,
    SET_A_PATH,
    SET_B_PATH,
    SET_C_PATH,
    SET_D_PATH,
    ensure_directories,
)

BASIC_STATS = ["mean", "std", "rms", "max_abs", "range"]
EXTENDED_STATS = ["median", "skew", "kurtosis", "crest_factor"]

RELATION_FEATURES = [
    "Vibration_total_rms",
    "Current_rms_to_Vibration_rms",
    "Current_mean_to_Vibration_rms",
    "Current_max_to_Vibration_rms",
    "AI0_to_AI1_rms_ratio",
    "Current_std_to_Vibration_std",
]

TEMPORAL_FEATURES = [
    "AI0_Vibration_slope",
    "AI0_Vibration_diff_rms",
    "AI1_Vibration_slope",
    "AI1_Vibration_diff_rms",
    "AI2_Current_slope",
    "AI2_Current_diff_rms",
    "AI0_AI1_corr",
]


def load_preprocessed_data():
    normal_df = pd.read_csv(NORMAL_PREPROCESSED_PATH, parse_dates=["TimeStamp"])
    abnormal_df = pd.read_csv(ABNORMAL_PREPROCESSED_PATH, parse_dates=["TimeStamp"])
    return normal_df, abnormal_df


def _sample_id(label: int, segment_id: int) -> str:
    return f"normal_{segment_id}" if label == 0 else f"abnormal_{segment_id}"


def extract_statistical_features(df: pd.DataFrame, label: int) -> pd.DataFrame:
    rows = []

    for segment_id, group in df.groupby("segment", sort=True):
        row = {
            "sample_id": _sample_id(label, segment_id),
            "segment": int(segment_id),
            "length": int(len(group)),
            "label": int(label),
        }

        for col in SENSOR_COLS:
            values = group[col].astype(float)
            mean = values.mean()
            std = values.std(ddof=0)
            rms = float(np.sqrt(np.mean(values.to_numpy() ** 2)))
            max_abs = values.abs().max()
            value_range = values.max() - values.min()
            median = values.median()
            skewness = values.skew()
            kurtosis = values.kurt()
            crest_factor = max_abs / rms if rms > 0 else 0.0

            row.update(
                {
                    f"{col}_mean": mean,
                    f"{col}_std": std,
                    f"{col}_rms": rms,
                    f"{col}_max_abs": max_abs,
                    f"{col}_range": value_range,
                    f"{col}_median": median,
                    f"{col}_skew": skewness,
                    f"{col}_kurtosis": kurtosis,
                    f"{col}_crest_factor": crest_factor,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_all_features(normal_df: pd.DataFrame, abnormal_df: pd.DataFrame) -> pd.DataFrame:
    normal_features = extract_statistical_features(normal_df, label=0)
    abnormal_features = extract_statistical_features(abnormal_df, label=1)
    return pd.concat([normal_features, abnormal_features], ignore_index=True)


def make_feature_set_a(feature_df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for sensor in SENSOR_COLS:
        cols.extend([f"{sensor}_{stat}" for stat in BASIC_STATS])
    return feature_df[["sample_id", "segment", "length", *cols, "label"]].copy()


def make_feature_set_b(feature_df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for sensor in SENSOR_COLS:
        cols.extend([f"{sensor}_{stat}" for stat in BASIC_STATS + EXTENDED_STATS])
    return feature_df[["sample_id", "segment", "length", *cols, "label"]].copy()


def make_feature_set_c(set_a: pd.DataFrame) -> pd.DataFrame:
    """Set A + vibration/current relationship features discovered during error analysis."""
    df = set_a.copy()

    df["Vibration_total_rms"] = np.sqrt(
        df["AI0_Vibration_rms"] ** 2 + df["AI1_Vibration_rms"] ** 2
    )
    df["Current_rms_to_Vibration_rms"] = (
        df["AI2_Current_rms"] / (df["Vibration_total_rms"] + EPS)
    )
    df["Current_mean_to_Vibration_rms"] = (
        np.abs(df["AI2_Current_mean"]) / (df["Vibration_total_rms"] + EPS)
    )
    df["Current_max_to_Vibration_rms"] = (
        df["AI2_Current_max_abs"] / (df["Vibration_total_rms"] + EPS)
    )
    df["AI0_to_AI1_rms_ratio"] = (
        df["AI0_Vibration_rms"] / (df["AI1_Vibration_rms"] + EPS)
    )

    vibration_total_std = np.sqrt(
        df["AI0_Vibration_std"] ** 2 + df["AI1_Vibration_std"] ** 2
    )
    df["Current_std_to_Vibration_std"] = (
        df["AI2_Current_std"] / (vibration_total_std + EPS)
    )
    return df


def calculate_temporal_features(group: pd.DataFrame) -> dict[str, float]:
    """Extract within-segment temporal shape features while preserving sample order."""
    group = group.sort_values("TimeStamp").reset_index(drop=True)
    elapsed_seconds = (
        group["TimeStamp"] - group["TimeStamp"].iloc[0]
    ).dt.total_seconds().to_numpy()

    row: dict[str, float] = {}
    for col in SENSOR_COLS:
        values = group[col].astype(float).to_numpy()

        if len(values) >= 2 and np.ptp(elapsed_seconds) > 0:
            slope = float(np.polyfit(elapsed_seconds, values, 1)[0])
        else:
            slope = np.nan

        if len(values) >= 2:
            diffs = np.diff(values)
            diff_rms = float(np.sqrt(np.mean(diffs ** 2)))
        else:
            diff_rms = np.nan

        row[f"{col}_slope"] = slope
        row[f"{col}_diff_rms"] = diff_rms

    ai0 = group["AI0_Vibration"].astype(float).to_numpy()
    ai1 = group["AI1_Vibration"].astype(float).to_numpy()

    if len(group) >= 3 and np.std(ai0) > 0 and np.std(ai1) > 0:
        corr = float(np.corrcoef(ai0, ai1)[0, 1])
    else:
        corr = np.nan
    row["AI0_AI1_corr"] = corr
    return row


def extract_temporal_features(df: pd.DataFrame, label: int) -> pd.DataFrame:
    rows = []
    for segment_id, group in df.groupby("segment", sort=True):
        row = {
            "sample_id": _sample_id(label, segment_id),
            "segment": int(segment_id),
            "length": int(len(group)),
            "label": int(label),
        }
        row.update(calculate_temporal_features(group))
        rows.append(row)
    return pd.DataFrame(rows)


def make_feature_set_d(
    set_c: pd.DataFrame,
    normal_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
) -> pd.DataFrame:
    normal_temporal = extract_temporal_features(normal_df, label=0)
    abnormal_temporal = extract_temporal_features(abnormal_df, label=1)
    temporal_df = pd.concat([normal_temporal, abnormal_temporal], ignore_index=True)

    if set(set_c["sample_id"]) != set(temporal_df["sample_id"]):
        raise ValueError("Set C와 시간 Feature의 sample_id가 일치하지 않습니다.")

    set_d = set_c.merge(
        temporal_df[["sample_id", *TEMPORAL_FEATURES]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if not set_c["sample_id"].equals(set_d["sample_id"]):
        raise ValueError("Set D 생성 과정에서 sample_id 순서가 변경되었습니다.")
    return set_d


def validate_feature_sets(feature_sets: dict[str, pd.DataFrame]) -> None:
    base = feature_sets["SetA"]
    for name, df in feature_sets.items():
        for col in META_COLS:
            if col not in df.columns:
                raise ValueError(f"{name}에 필수 컬럼 {col}이 없습니다.")
        if not base["sample_id"].equals(df["sample_id"]):
            raise ValueError(f"Set A와 {name}의 sample_id 순서가 다릅니다.")
        if not base["label"].equals(df["label"]):
            raise ValueError(f"Set A와 {name}의 label 순서가 다릅니다.")

        numeric = df.select_dtypes(include=[np.number])
        if np.isinf(numeric).any().any():
            raise ValueError(f"{name}에 무한대 값이 있습니다.")


def run_feature_engineering():
    ensure_directories()
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING START")
    print("=" * 60)

    normal_df, abnormal_df = load_preprocessed_data()
    feature_df = build_all_features(normal_df, abnormal_df)

    set_a = make_feature_set_a(feature_df)
    set_b = make_feature_set_b(feature_df)
    set_c = make_feature_set_c(set_a)
    set_d = make_feature_set_d(set_c, normal_df, abnormal_df)

    feature_sets = {"SetA": set_a, "SetB": set_b, "SetC": set_c, "SetD": set_d}
    validate_feature_sets(feature_sets)

    feature_df.to_csv(ALL_FEATURES_PATH, index=False)
    set_a.to_csv(SET_A_PATH, index=False)
    set_b.to_csv(SET_B_PATH, index=False)
    set_c.to_csv(SET_C_PATH, index=False)
    set_d.to_csv(SET_D_PATH, index=False)

    print(f"전체 Feature: {feature_df.shape}")
    for name, df in feature_sets.items():
        print(f"{name}: shape={df.shape}, 모델 입력={df.shape[1] - 4}")
    print("Feature 파일 저장 완료")
    return feature_sets


if __name__ == "__main__":
    run_feature_engineering()
