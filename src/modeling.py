from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    META_COLS,
    MODELING_DIR,
    N_SPLITS,
    RANDOM_STATE,
    RESULT_DIR,
    SET_A_PATH,
    SET_B_PATH,
    SET_C_PATH,
    SET_D_PATH,
    THRESHOLD,
    ensure_directories,
)

FEATURE_PATHS = {
    "SetA": SET_A_PATH,
    "SetB": SET_B_PATH,
    "SetC": SET_C_PATH,
    "SetD": SET_D_PATH,
}


def load_feature_sets() -> dict[str, pd.DataFrame]:
    feature_sets = {name: pd.read_csv(path) for name, path in FEATURE_PATHS.items()}
    base = feature_sets["SetA"]
    for name, df in feature_sets.items():
        if not base["sample_id"].equals(df["sample_id"]):
            raise ValueError(f"Set A와 {name}의 sample_id 순서가 다릅니다.")
        if not base["label"].equals(df["label"]):
            raise ValueError(f"Set A와 {name}의 label 순서가 다릅니다.")
    return feature_sets


def split_xy(df: pd.DataFrame):
    feature_cols = [col for col in df.columns if col not in META_COLS]
    X = df[feature_cols].copy()
    y = df["label"].astype(int).copy()
    meta = df[["sample_id", "segment", "length"]].copy()
    return X, y, meta, feature_cols


def get_models() -> dict[str, Pipeline]:
    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {"LogisticRegression": logistic, "RandomForest": random_forest}


def make_cv_splits(y: pd.Series):
    if int((y == 1).sum()) < N_SPLITS or int((y == 0).sum()) < N_SPLITS:
        raise ValueError(f"{N_SPLITS}-Fold를 수행하기에 각 클래스 샘플 수가 부족합니다.")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return list(cv.split(np.zeros(len(y)), y))


def calculate_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }


def _save_model_interpretation(
    fitted_pipeline: Pipeline,
    feature_cols: list[str],
    model_name: str,
    fold_no: int,
) -> pd.DataFrame | None:
    estimator = fitted_pipeline.named_steps["model"]
    if model_name == "RandomForest":
        values = estimator.feature_importances_
        value_name = "importance"
    elif model_name == "LogisticRegression":
        values = estimator.coef_[0]
        value_name = "coefficient"
    else:
        return None

    return pd.DataFrame(
        {
            "fold": fold_no,
            "feature": feature_cols,
            value_name: values,
        }
    )


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    feature_cols: list[str],
    model: Pipeline,
    model_name: str,
    feature_set_name: str,
    splits,
) -> dict[str, float]:
    oof_prob = np.zeros(len(y), dtype=float)
    oof_pred = np.zeros(len(y), dtype=int)
    fold_number = np.zeros(len(y), dtype=int)
    fold_results = []
    interpretation_frames = []

    for fold_no, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        val_prob = fold_model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= THRESHOLD).astype(int)

        oof_prob[val_idx] = val_prob
        oof_pred[val_idx] = val_pred
        fold_number[val_idx] = fold_no

        metrics = calculate_metrics(y_val, val_pred, val_prob)
        metrics["fold"] = fold_no
        fold_results.append(metrics)

        interp = _save_model_interpretation(
            fold_model, feature_cols, model_name, fold_no
        )
        if interp is not None:
            interpretation_frames.append(interp)

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(
        MODELING_DIR / f"{feature_set_name}_{model_name}_fold_metrics.csv",
        index=False,
    )

    metric_cols = [
        "precision",
        "recall",
        "f1",
        "accuracy",
        "pr_auc",
        "roc_auc",
        "fpr",
        "fnr",
    ]
    fold_mean = fold_df[metric_cols].mean()
    fold_std = fold_df[metric_cols].std()
    oof_metrics = calculate_metrics(y, oof_pred, oof_prob)

    prediction_df = meta.copy()
    prediction_df["actual"] = y.values
    prediction_df["probability"] = oof_prob
    prediction_df["prediction"] = oof_pred
    prediction_df["fold"] = fold_number
    prediction_df["result_type"] = np.select(
        [
            (prediction_df["actual"] == 0) & (prediction_df["prediction"] == 0),
            (prediction_df["actual"] == 0) & (prediction_df["prediction"] == 1),
            (prediction_df["actual"] == 1) & (prediction_df["prediction"] == 0),
            (prediction_df["actual"] == 1) & (prediction_df["prediction"] == 1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="UNKNOWN",
    )

    pred_path = MODELING_DIR / f"{feature_set_name}_{model_name}_predictions.csv"
    prediction_df.to_csv(pred_path, index=False)
    prediction_df[prediction_df["result_type"].isin(["FP", "FN"])].to_csv(
        MODELING_DIR / f"{feature_set_name}_{model_name}_errors.csv",
        index=False,
    )

    if interpretation_frames:
        interpretation = pd.concat(interpretation_frames, ignore_index=True)
        interpretation.to_csv(
            MODELING_DIR / f"{feature_set_name}_{model_name}_fold_interpretation.csv",
            index=False,
        )

    result = {
        "feature_set": feature_set_name,
        "model": model_name,
        "feature_count": int(X.shape[1]),
        **{key: oof_metrics[key] for key in ["TN", "FP", "FN", "TP"]},
        **{f"{m}_oof": oof_metrics[m] for m in metric_cols},
    }
    for metric in ["precision", "recall", "f1", "pr_auc", "roc_auc"]:
        result[f"{metric}_mean"] = float(fold_mean[metric])
        result[f"{metric}_std"] = float(fold_std[metric])
    return result


def run_modeling():
    ensure_directories()
    print("\n" + "=" * 60)
    print("MODELING START")
    print("=" * 60)

    feature_sets = load_feature_sets()
    y_reference = feature_sets["SetA"]["label"].astype(int)
    splits = make_cv_splits(y_reference)

    for fold_no, (train_idx, val_idx) in enumerate(splits, start=1):
        train_y = y_reference.iloc[train_idx]
        val_y = y_reference.iloc[val_idx]
        print(
            f"Fold {fold_no}: Train 정상={(train_y==0).sum()}, 이상={(train_y==1).sum()} | "
            f"Validation 정상={(val_y==0).sum()}, 이상={(val_y==1).sum()}"
        )

    all_results = []
    models = get_models()

    for set_name, feature_df in feature_sets.items():
        X, y, meta, feature_cols = split_xy(feature_df)
        for model_name, model in models.items():
            result = evaluate_model(
                X=X,
                y=y,
                meta=meta,
                feature_cols=feature_cols,
                model=model,
                model_name=model_name,
                feature_set_name=set_name,
                splits=splits,
            )
            all_results.append(result)
            print(
                f"{set_name} + {model_name}: "
                f"FP={result['FP']}, FN={result['FN']}, TP={result['TP']}, "
                f"Recall={result['recall_oof']:.4f}, F1={result['f1_oof']:.4f}, "
                f"PR-AUC={result['pr_auc_oof']:.4f}"
            )

    comparison_df = pd.DataFrame(all_results).sort_values(
        by=["f1_oof", "pr_auc_oof"], ascending=[False, False]
    ).reset_index(drop=True)
    comparison_df.to_csv(MODELING_DIR / "model_comparison.csv", index=False)

    comparison_df[comparison_df["model"] == "RandomForest"].to_csv(
        MODELING_DIR / "random_forest_feature_set_comparison.csv", index=False
    )
    comparison_df[comparison_df["model"] == "LogisticRegression"].to_csv(
        MODELING_DIR / "logistic_feature_set_comparison.csv", index=False
    )

    # 별도 독립 test.csv가 제공되지 않은 경우 제출/검증용 예측결과로 OOF를 사용한다.
    final_oof_source = MODELING_DIR / "SetD_LogisticRegression_predictions.csv"
    if final_oof_source.exists():
        shutil.copyfile(final_oof_source, RESULT_DIR / "oof_predictions.csv")

    print("\nFINAL MODEL COMPARISON")
    show_cols = [
        "feature_set", "model", "feature_count", "TN", "FP", "FN", "TP",
        "precision_oof", "recall_oof", "f1_oof", "pr_auc_oof", "roc_auc_oof",
        "f1_mean", "f1_std", "recall_mean", "recall_std",
    ]
    print(comparison_df[show_cols].round(4).to_string(index=False))
    print("MODELING COMPLETE")
    return comparison_df


if __name__ == "__main__":
    run_modeling()
