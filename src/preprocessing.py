from __future__ import annotations

import pandas as pd

from .config import (
    ABNORMAL_PREPROCESSED_PATH,
    ABNORMAL_RAW_PATH,
    GAP_SECONDS,
    NORMAL_PREPROCESSED_PATH,
    NORMAL_RAW_PATH,
    REQUIRED_COLUMNS,
    ensure_directories,
)


def load_raw_data(path):
    return pd.read_csv(path)


def basic_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove CSV index artifacts/duplicate rows and standardize timestamp order."""
    df = df.copy()

    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="raise")

    before = len(df)
    df = df.drop_duplicates().copy()
    removed_duplicates = before - len(df)

    df = df.sort_values("TimeStamp").reset_index(drop=True)
    return df, removed_duplicates


def add_segments(df: pd.DataFrame, gap_seconds: float = GAP_SECONDS) -> pd.DataFrame:
    """Split independent continuous segments when the timestamp gap exceeds the threshold."""
    df = df.copy()
    time_diff = df["TimeStamp"].diff()
    new_segment = time_diff > pd.Timedelta(seconds=gap_seconds)
    df["segment"] = new_segment.cumsum().astype(int)
    return df


def validate_preprocessed_data(
    df: pd.DataFrame,
    dataset_name: str,
    expected_label: int,
) -> None:
    label_values = df["Equipment_state"].dropna().unique()
    if len(label_values) == 0 or not all(value == expected_label for value in label_values):
        raise ValueError(
            f"{dataset_name}: 예상하지 않은 Equipment_state가 있습니다: {label_values}"
        )

    if df[REQUIRED_COLUMNS].isna().any().any():
        missing = df[REQUIRED_COLUMNS].isna().sum()
        raise ValueError(f"{dataset_name}: 필수 변수에 결측치가 있습니다.\n{missing[missing > 0]}")

    lengths = df.groupby("segment").size()
    print(f"\n[{dataset_name}] shape={df.shape}, segments={df['segment'].nunique()}")
    print(f"중복 행={df.duplicated().sum()}, 길이<10 Segment={(lengths < 10).sum()}")
    print(lengths.describe().round(3).to_string())


def preprocess_dataset(
    input_path,
    output_path,
    dataset_name: str,
    expected_label: int,
    gap_seconds: float = GAP_SECONDS,
) -> pd.DataFrame:
    df = load_raw_data(input_path)
    print(f"\n{dataset_name} 원본 shape: {df.shape}")

    df, removed_duplicates = basic_cleaning(df)
    print(f"{dataset_name} 제거된 중복 행: {removed_duplicates}")

    df = add_segments(df, gap_seconds=gap_seconds)
    validate_preprocessed_data(df, dataset_name, expected_label)

    df.to_csv(output_path, index=False)
    print(f"저장 완료: {output_path}")
    return df


def run_preprocessing(gap_seconds: float = GAP_SECONDS):
    """Run preprocessing directly from the two organizer-provided raw CSV files."""
    ensure_directories()
    print("\n" + "=" * 60)
    print("PREPROCESSING START")
    print("=" * 60)

    if not NORMAL_RAW_PATH.exists() or not ABNORMAL_RAW_PATH.exists():
        missing = [
            str(p.name)
            for p in [NORMAL_RAW_PATH, ABNORMAL_RAW_PATH]
            if not p.exists()
        ]
        raise FileNotFoundError(
            "data 폴더에 원본 CSV가 필요합니다: " + ", ".join(missing)
        )

    normal_df = preprocess_dataset(
        NORMAL_RAW_PATH,
        NORMAL_PREPROCESSED_PATH,
        "normal",
        expected_label=0,
        gap_seconds=gap_seconds,
    )
    abnormal_df = preprocess_dataset(
        ABNORMAL_RAW_PATH,
        ABNORMAL_PREPROCESSED_PATH,
        "abnormal",
        expected_label=1,
        gap_seconds=gap_seconds,
    )

    print("\nPREPROCESSING COMPLETE")
    print(f"Normal Segment: {normal_df['segment'].nunique()}")
    print(f"Abnormal Segment: {abnormal_df['segment'].nunique()}")
    return normal_df, abnormal_df


if __name__ == "__main__":
    run_preprocessing()
