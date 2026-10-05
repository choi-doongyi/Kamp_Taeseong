from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = PROJECT_ROOT / "figures"

NORMAL_RAW_PATH = DATA_DIR / "press_data_normal.csv"
ABNORMAL_RAW_PATH = DATA_DIR / "outlier_data.csv"
NORMAL_PREPROCESSED_PATH = DATA_DIR / "normal_preprocessed.csv"
ABNORMAL_PREPROCESSED_PATH = DATA_DIR / "outlier_preprocessed.csv"

ALL_FEATURES_PATH = RESULT_DIR / "all_features.csv"
SET_A_PATH = RESULT_DIR / "feature_set_a.csv"
SET_B_PATH = RESULT_DIR / "feature_set_b.csv"
SET_C_PATH = RESULT_DIR / "feature_set_c.csv"
SET_D_PATH = RESULT_DIR / "feature_set_d.csv"

MODELING_DIR = RESULT_DIR / "modeling"
ALARM_DIR = RESULT_DIR / "two_stage_alarm"
ANALYSIS_DIR = RESULT_DIR / "analysis"
GAP_DIR = RESULT_DIR / "gap_sensitivity"

SENSOR_COLS = [
    "AI0_Vibration",
    "AI1_Vibration",
    "AI2_Current",
]

REQUIRED_COLUMNS = [
    "TimeStamp",
    *SENSOR_COLS,
    "Equipment_state",
]

META_COLS = ["sample_id", "segment", "length", "label"]

GAP_SECONDS = 0.5
RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5
EPS = 1e-6


def ensure_directories() -> None:
    for path in [DATA_DIR, RESULT_DIR, FIGURE_DIR, MODELING_DIR, ALARM_DIR, ANALYSIS_DIR, GAP_DIR]:
        path.mkdir(parents=True, exist_ok=True)
