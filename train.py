"""Train, tune, evaluate and save a prediction model.

Examples:
    python train.py                                          # bundled breast-cancer demo
    python train.py --csv data/sample_churn.csv --target churn --drop customer_id
    python train.py --csv data/sample_house_prices.csv --target price --drop listing_id
    python train.py --csv mydata.csv --target label --quick  # skip the tuning grid (faster)
    python train.py --csv panel.csv --target y --time-col year --gap 1   # forecasting: no peeking

The task (classification or regression) is detected from the target column; override
it with --task. Categorical columns and missing values are handled automatically.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import build_schema, column_spec, default_model_dir, prepare_input


# --------------------------------------------------------------------------- helpers
def _native(v):
    return v.item() if hasattr(v, "item") else v


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Cannot serialise {type(o)}")


def detect_task(y: pd.Series, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if (not pd.api.types.is_numeric_dtype(y)) or pd.api.types.is_bool_dtype(y):
        return "classification"
    integer_valued = bool(np.all(np.mod(y.dropna().to_numpy(dtype=float), 1) == 0))
    return "classification" if (y.nunique() <= 10 and integer_valued) else "regression"


def build_preprocessor(schema: dict) -> ColumnTransformer:
    numeric = [c for c, s in schema.items() if s["type"] == "numeric"]
    categorical = [c for c, s in schema.items() if s["type"] == "categorical"]
    parts = []
    if numeric:
        parts.append(
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        parts.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(parts)


def get_candidates(task: str, seed: int) -> dict:
    """name -> (estimator, hyper-parameter grid). Grid keys are prefixed with `model__`."""
    if task == "classification":
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=3000),
                {"model__C": [0.1, 1.0, 10.0]},
            ),
            "random_forest": (
                RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
                {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 3]},
            ),
            "hist_gradient_boosting": (
                HistGradientBoostingClassifier(random_state=seed),
                {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 5]},
            ),
        }
    return {
        "ridge": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0]}),
        "random_forest": (
            RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
            {"model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(random_state=seed),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 5]},
        ),
    }


def evaluate(best, X_test, y_test, task: str):
    """Return (metrics, details, predictions) on the held-out test set."""
    pred = best.predict(X_test)
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        metrics = {
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }
        return {k: round(v, 4) for k, v in metrics.items()}, {}, pred

    classes = [_native(c) for c in best.classes_]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
    }
    proba = best.predict_proba(X_test)
    try:
        if len(classes) == 2:
            metrics["roc_auc"] = float(roc_auc_score((y_test == classes[1]).astype(int), proba[:, 1]))
        else:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_test, proba, multi_class="ovr", labels=classes)
            )
    except ValueError:
        pass  # e.g. a class missing from the test split
    details = {
        "labels": [str(c) for c in classes],
        "confusion_matrix": confusion_matrix(y_test, pred, labels=classes).tolist(),
        "report": classification_report(y_test, pred, zero_division=0),
    }
    return {k: round(v, 4) for k, v in metrics.items()}, details, pred


def save_plots(report_dir: Path, task: str, importance, details: dict, y_test, pred) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return
    try:
        if importance is not None and not importance.empty:
            top = importance.head(15)[::-1]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
            ax.set_xlabel("Drop in score when the feature is shuffled")
            ax.set_title("Permutation feature importance (test set)")
            fig.tight_layout()
            fig.savefig(report_dir / "feature_importance.png", dpi=130)
            plt.close(fig)
        if task == "classification":
            cm = np.array(details["confusion_matrix"])
            labels = details["labels"]
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(labels)), labels)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion matrix (test set)")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, int(v), ha="center", va="center")
            fig.tight_layout()
            fig.savefig(report_dir / "confusion_matrix.png", dpi=130)
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(y_test, pred, s=8, alpha=0.5)
            lo, hi = float(min(np.min(y_test), np.min(pred))), float(max(np.max(y_test), np.max(pred)))
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Predicted vs actual (test set)")
            fig.tight_layout()
            fig.savefig(report_dir / "predicted_vs_actual.png", dpi=130)
            plt.close(fig)
    except Exception as err:  # plots are a nice-to-have, never fail training over them
        print(f"Could not draw plots: {err}")


def chronological_split(time_values, test_size: float, gap: int):
    """Split row positions by time: oldest periods train, newest periods test.

    `gap` whole periods between them are discarded. Use gap >= the forecast horizon so that no
    training target overlaps the test period (otherwise the model peeks at the future).
    """
    t = np.asarray(time_values)
    periods = np.sort(pd.unique(t))
    n_test = max(1, int(round(len(periods) * test_size)))
    n_train = len(periods) - n_test - gap
    if n_train < 3:
        raise ValueError(
            f"Only {len(periods)} distinct time periods - not enough for a chronological split "
            f"with a {gap}-period gap. Add history or lower --test-size / --gap."
        )
    train_periods, test_periods = periods[:n_train], periods[-n_test:]
    idx_train = np.flatnonzero(np.isin(t, train_periods))
    idx_train = idx_train[np.argsort(t[idx_train], kind="stable")]  # TimeSeriesSplit needs time order
    idx_test = np.flatnonzero(np.isin(t, test_periods))
    info = {
        "type": "chronological",
        "gap_periods": int(gap),
        "n_train_periods": int(n_train),
        "train_periods": [_native(train_periods[0]), _native(train_periods[-1])],
        "test_periods": [_native(test_periods[0]), _native(test_periods[-1])],
    }
    return idx_train, idx_test, info


# --------------------------------------------------------------------------- training
def train(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: str = "auto",
    model_dir=None,
    quick: bool = False,
    test_size: float = 0.2,
    seed: int = 42,
    target_name: str = "target",
    data_source: str = "custom data",
    verbose: bool = True,
    time_values=None,
    gap: int = 0,
    extra: pd.DataFrame | None = None,
) -> dict:
    """Tune several models, pick the best by cross-validation, evaluate, and save everything.

    time_values: optional per-row time (sortable numbers, e.g. years). When given, the split is
        chronological and cross-validation respects time order - use it for any forecasting task.
    gap: periods discarded between train and test (set to the forecast horizon).
    extra: optional identifier columns (not used as features) saved next to the test predictions.
    """
    say = print if verbose else (lambda *a, **k: None)
    model_dir = Path(model_dir) if model_dir else default_model_dir()
    report_dir = model_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    keep = y.notna().to_numpy()
    if not keep.all():
        say(f"Dropping {int((~keep).sum())} rows with a missing target")
    pos = np.flatnonzero(keep)
    X, y = X.iloc[pos].reset_index(drop=True), y.iloc[pos].reset_index(drop=True)
    if time_values is not None:
        time_values = pd.Series(np.asarray(time_values)[pos], name=getattr(time_values, "name", None) or "period")
    if extra is not None:
        extra = extra.iloc[pos].reset_index(drop=True)
    if len(X) < 20:
        raise ValueError("Need at least 20 rows with a target value to train.")

    task = detect_task(y, task)
    if task == "classification" and y.nunique() < 2:
        raise ValueError("The target has only one class - nothing to predict.")
    say(f"Task: {task} | rows: {len(X)} | features: {X.shape[1]} | source: {data_source}")

    if time_values is None:
        stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify
        )
        split_info = {"type": "random"}
    else:
        idx_train, idx_test, split_info = chronological_split(time_values, test_size, gap)
        X_train_raw, X_test_raw = X.iloc[idx_train], X.iloc[idx_test]
        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
        say(
            f"Chronological split: train {split_info['train_periods']}, "
            f"gap {gap}, test {split_info['test_periods']}"
        )
    schema = build_schema(X_train_raw)  # built from training rows only
    X_train = prepare_input(X_train_raw, schema)
    X_test = prepare_input(X_test_raw, schema)

    if time_values is not None:
        rows_per_period = int(np.ceil(len(X_train) / split_info["n_train_periods"]))
        cv = TimeSeriesSplit(
            n_splits=max(2, min(4, split_info["n_train_periods"] - 2)), gap=gap * rows_per_period
        )
    if task == "classification":
        if time_values is None:
            n_splits = max(2, min(5, int(y_train.value_counts().min())))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scoring = "roc_auc" if y_train.nunique() == 2 else "f1_macro"
        score_label = "ROC-AUC" if scoring == "roc_auc" else "F1-macro"
    else:
        if time_values is None:
            cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        scoring = "neg_root_mean_squared_error"
        score_label = "RMSE"
    sign = -1.0 if scoring.startswith("neg_") else 1.0  # show RMSE as a positive number

    # Dummy model: the score any real model has to beat.
    dummy = DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor()
    dummy_pipe = Pipeline([("prep", build_preprocessor(schema)), ("model", dummy)])
    dummy_score = sign * float(cross_val_score(dummy_pipe, X_train, y_train, cv=cv, scoring=scoring).mean())

    rows, best_search, best_name = [], None, None
    for name, (estimator, grid) in get_candidates(task, seed).items():
        pipe = Pipeline([("prep", build_preprocessor(schema)), ("model", estimator)])
        t0 = time.time()
        search = GridSearchCV(pipe, {} if quick else grid, cv=cv, scoring=scoring, refit=True)
        search.fit(X_train, y_train)
        params = {k.replace("model__", ""): _native(v) for k, v in search.best_params_.items()}
        rows.append(
            {
                "model": name,
                f"cv_{score_label}": round(sign * float(search.best_score_), 4),
                "best_params": json.dumps(params, default=_json_default),
                "seconds": round(time.time() - t0, 1),
            }
        )
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_search, best_name = search, name

    comparison = pd.DataFrame(rows)
    comparison["selected"] = comparison["model"] == best_name
    comparison.loc[len(comparison)] = ["baseline (dummy)", round(dummy_score, 4), "{}", 0.0, False]
    say(f"\nModel comparison ({score_label}, {cv.get_n_splits()}-fold CV on training rows):")
    say(comparison.to_string(index=False))
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)

    best = best_search.best_estimator_
    best_params = {k.replace("model__", ""): _native(v) for k, v in best_search.best_params_.items()}
    say(f"\nSelected: {best_name} {best_params}")

    metrics, details, pred = evaluate(best, X_test, y_test, task)
    say("\nHeld-out test set: " + ", ".join(f"{k}={v}" for k, v in metrics.items()))
    if task == "classification":
        say(details["report"])
        (report_dir / "classification_report.txt").write_text(details["report"])

    test_predictions = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": pred}, index=X_test_raw.index)
    if extra is not None:
        test_predictions = pd.concat([extra.loc[X_test_raw.index], test_predictions], axis=1)
    if time_values is not None:
        test_predictions.insert(0, time_values.name, time_values.loc[X_test_raw.index].to_numpy())
    test_predictions.to_csv(report_dir / "test_predictions.csv", index=False)

    importance = None
    try:
        sample = X_test.sample(min(len(X_test), 2000), random_state=seed)
        result = permutation_importance(
            best, sample, y_test.loc[sample.index], scoring=scoring, n_repeats=5, random_state=seed
        )
        importance = (
            pd.DataFrame(
                {
                    "feature": X_test.columns,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        importance.to_csv(report_dir / "feature_importance.csv", index=False)
        say("Top features (permutation importance):")
        say(importance.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    except Exception as err:
        say(f"Could not compute feature importance: {err}")

    save_plots(report_dir, task, importance, details, y_test, pred)

    holdout = X_test_raw.head(200).copy()
    holdout[target_name] = y_test.head(200).to_numpy()
    holdout.to_csv(report_dir / "holdout_sample.csv", index=False)

    classes = [_native(c) for c in best.classes_] if task == "classification" else None
    meta = {
        "task": task,
        "target": target_name,
        "model": best_name,
        "best_params": best_params,
        "scoring": scoring,
        "cv_score": round(sign * float(best_search.best_score_), 4),
        "cv_score_label": score_label,
        "baseline_cv_score": round(dummy_score, 4),
        "test_metrics": metrics,
        "classes": classes,
        "positive_class": classes[1] if classes and len(classes) == 2 else None,
        "features": list(X.columns),
        "schema": schema,
        # How the model's own predictions were distributed - used for prediction-drift checks.
        "baseline": column_spec(pd.Series(pred), categorical=(task == "classification")),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split": split_info,
        "data_source": data_source,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(best, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    say(f"\nSaved model + reports to {model_dir}/")
    return meta


# --------------------------------------------------------------------------- CLI
def _split_names(text: str) -> list:
    return [c.strip() for c in (text or "").split(",") if c.strip()]


def load_dataset(csv, target: str, drop: str, time_col: str = "", keep_cols: str = ""):
    """Return (X, y, target_name, source, time_values, extra)."""
    if not csv:
        data = load_breast_cancer(as_frame=True)
        return data.data, data.target, "target", "scikit-learn breast cancer demo dataset", None, None
    df = pd.read_csv(csv)
    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found. Columns are: {list(df.columns)}")
    drop_cols, keep = _split_names(drop), _split_names(keep_cols)
    unknown = [c for c in drop_cols + keep + _split_names(time_col) if c not in df.columns]
    if unknown:
        raise SystemExit(f"Columns not found in the CSV: {unknown}")
    time_values = df[time_col] if time_col else None
    extra = df[keep].copy() if keep else None
    X = df.drop(columns=[target] + drop_cols + keep + ([time_col] if time_col else []))
    for col in X.select_dtypes(exclude="number").columns:
        if len(X) > 50 and X[col].nunique() > 0.5 * len(X):
            print(f"Warning: '{col}' looks like an ID (almost every value is unique). "
                  f"Consider --drop {col}")
    return X, df[target], target, str(csv), time_values, extra


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="CSV file with feature columns and a target column")
    parser.add_argument("--target", default="target", help="Name of the target column (default: target)")
    parser.add_argument("--drop", default="", help="Comma-separated columns to ignore, e.g. IDs")
    parser.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    parser.add_argument("--quick", action="store_true", help="Skip hyper-parameter search (much faster)")
    parser.add_argument("--model-dir", help="Where to save the model (default: ./model)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-col", default="", help="Time column: switches to a chronological split")
    parser.add_argument("--gap", type=int, default=0, help="Periods dropped between train and test")
    parser.add_argument("--keep-cols", default="", help="ID columns kept out of the features but saved "
                                                         "beside the test predictions")
    args = parser.parse_args()

    X, y, target_name, source, time_values, extra = load_dataset(
        args.csv, args.target, args.drop, args.time_col, args.keep_cols
    )
    try:
        train(
            X, y, task=args.task, model_dir=args.model_dir, quick=args.quick,
            test_size=args.test_size, seed=args.seed, target_name=target_name, data_source=source,
            time_values=time_values, gap=args.gap, extra=extra,
        )
    except ValueError as err:
        raise SystemExit(f"Error: {err}")


if __name__ == "__main__":
    main()
