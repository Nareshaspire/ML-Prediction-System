"""Shared helpers used by training, prediction, the API, the web app and monitoring."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OTHER = "__other__"  # bucket for rare / unseen categories


# --------------------------------------------------------------------------- paths
def default_model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", BASE_DIR / "model"))


def default_log_dir() -> Path:
    return Path(os.environ.get("LOG_DIR", BASE_DIR / "logs"))


def load_metadata(model_dir=None) -> dict:
    model_dir = Path(model_dir) if model_dir else default_model_dir()
    path = model_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found in '{model_dir}'. Run `python train.py` first."
        )
    return json.loads(path.read_text())


def load_model(model_dir=None):
    """Return (pipeline, metadata) for the saved model."""
    model_dir = Path(model_dir) if model_dir else default_model_dir()
    meta = load_metadata(model_dir)
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found in '{model_dir}'. Run `python train.py` first."
        )
    return joblib.load(model_path), meta


# --------------------------------------------------------------------------- schema
def column_spec(series: pd.Series, categorical: bool | None = None) -> dict:
    """Describe one column and how it was distributed in training.

    The description is used for input validation, the web form, and drift monitoring.
    """
    if categorical is None:
        categorical = (not pd.api.types.is_numeric_dtype(series)) or pd.api.types.is_bool_dtype(series)

    if categorical:
        shares = series.dropna().astype(str).value_counts(normalize=True)
        top = shares.head(20)
        expected = {str(k): float(v) for k, v in top.items()}
        rest = 1.0 - float(top.sum())
        if rest > 1e-9:
            expected[OTHER] = rest
        return {"type": "categorical", "choices": [str(k) for k in top.index], "expected": expected}

    values = pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()
    if values.size == 0:
        return {"type": "numeric", "min": 0.0, "max": 0.0, "median": 0.0, "edges": [], "expected": [1.0]}
    edges = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
    counts = np.bincount(np.digitize(values, edges), minlength=len(edges) + 1)
    return {
        "type": "numeric",
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "edges": edges.tolist(),
        "expected": (counts / counts.sum()).tolist(),
    }


def build_schema(X: pd.DataFrame) -> dict:
    return {col: column_spec(X[col]) for col in X.columns}


def prepare_input(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Validate and normalise raw rows so they look exactly like the training data."""
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = pd.DataFrame(index=df.index)
    for col, spec in schema.items():
        s = df[col]
        if spec["type"] == "numeric":
            out[col] = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            out[col] = s.map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    return out


# --------------------------------------------------------------------------- prediction
def predict_frame(pipeline, meta: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Score rows. Returns a frame with `prediction` and, for classifiers, a probability."""
    X = prepare_input(df, meta["schema"])
    out = pd.DataFrame({"prediction": pipeline.predict(X)}, index=df.index)
    if meta["task"] == "classification":
        proba = pipeline.predict_proba(X)
        if proba.shape[1] == 2:
            out["probability"] = proba[:, 1].round(4)  # P(positive_class)
        else:
            out["confidence"] = proba.max(axis=1).round(4)
    return out


# --------------------------------------------------------------------------- logging
def log_predictions(features: pd.DataFrame, result: pd.DataFrame, source: str, log_dir=None) -> int:
    """Append scored rows to logs/predictions.jsonl so drift can be monitored later."""
    log_dir = Path(log_dir) if log_dir else default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(features.to_json(orient="records"))
    outputs = json.loads(result.to_json(orient="records"))
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(log_dir / "predictions.jsonl", "a", encoding="utf-8") as fh:
        for inp, out in zip(inputs, outputs):
            fh.write(json.dumps({"ts": ts, "source": source, "inputs": inp, "output": out}) + "\n")
    return len(inputs)


def read_logs(log_dir=None):
    """Return (inputs, outputs) DataFrames from the prediction log (empty if none)."""
    path = (Path(log_dir) if log_dir else default_log_dir()) / "predictions.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    inputs = pd.DataFrame([r.get("inputs", {}) for r in rows])
    outputs = pd.DataFrame(
        [{**r.get("output", {}), "ts": r.get("ts"), "source": r.get("source")} for r in rows]
    )
    return inputs, outputs
