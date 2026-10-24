"""Check logged predictions for data drift.

    python monitor.py                      # report on logs/predictions.jsonl
    python monitor.py --fail-on-drift      # exit code 1 if anything drifted (for cron / CI)

Compares what the model is being asked to score *now* with what it saw in training, using
the Population Stability Index (PSI) per feature and on the predictions themselves:

    PSI < 0.10   ok      distribution is stable
    0.10 - 0.25  watch   moderate shift
    PSI > 0.25   drift   significant shift - investigate, consider retraining

PSI is noisy with few rows, so a report needs at least --min-rows logged predictions.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from common import OTHER, load_metadata, read_logs

WATCH, DRIFT = 0.10, 0.25


def psi(expected, actual, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if value < WATCH:
        return "ok"
    return "watch" if value < DRIFT else "drift"


def column_drift(spec: dict, values):
    """PSI between the training distribution described by `spec` and the observed `values`."""
    if spec["type"] == "numeric":
        vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            return None
        edges = np.asarray(spec["edges"], dtype=float)
        counts = np.bincount(np.digitize(vals, edges), minlength=len(edges) + 1)
        return psi(spec["expected"], counts / counts.sum())

    expected = spec["expected"]
    keys = list(expected) + ([] if OTHER in expected else [OTHER])
    s = pd.Series(values).dropna().astype(str)
    if s.empty:
        return None
    s = s.where(s.isin(list(expected)), OTHER)
    shares = s.value_counts(normalize=True)
    return psi([expected.get(k, 0.0) for k in keys], [float(shares.get(k, 0.0)) for k in keys])


def compute_drift(meta: dict, inputs: pd.DataFrame, outputs: pd.DataFrame):
    """Return (per-feature drift table, prediction-drift dict or None)."""
    rows = []
    for col, spec in meta["schema"].items():
        if col in inputs.columns:
            value = column_drift(spec, inputs[col])
            rows.append({"feature": col, "psi": value, "status": status(value)})
    features = pd.DataFrame(rows, columns=["feature", "psi", "status"])
    features = features.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)

    prediction = None
    if "prediction" in outputs.columns and meta.get("baseline"):
        value = column_drift(meta["baseline"], outputs["prediction"])
        prediction = {"psi": value, "status": status(value)}
    return features, prediction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", help="Model folder (default: ./model)")
    parser.add_argument("--log-dir", help="Log folder (default: ./logs)")
    parser.add_argument("--min-rows", type=int, default=30, help="Minimum logged rows for a report")
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit with code 1 if drift is found")
    args = parser.parse_args()

    try:
        meta = load_metadata(args.model_dir)
    except FileNotFoundError as err:
        print(err)
        return 2

    inputs, outputs = read_logs(args.log_dir)
    if len(inputs) < args.min_rows:
        print(
            f"Only {len(inputs)} logged predictions; need at least {args.min_rows} for a "
            "meaningful report. Score more rows with predict.py, the API or the web app."
        )
        return 0

    sources = outputs["source"].value_counts().to_dict()
    print(f"Monitoring report: {len(inputs)} logged predictions {sources}")
    print(f"Model: {meta['model']} trained {meta['trained_at']}\n")

    features, prediction = compute_drift(meta, inputs, outputs)
    print("Feature drift (PSI vs training data):")
    print(features.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if prediction is not None and prediction["psi"] is not None:
        print(f"\nPrediction drift: PSI={prediction['psi']:.3f} ({prediction['status']})")
    print("\nPSI < 0.10 ok | 0.10-0.25 watch | > 0.25 drift")

    drifted = (features["status"] == "drift").any() or (prediction or {}).get("status") == "drift"
    if drifted:
        print("\nDrift detected - the model may be seeing data unlike its training set.")
    return 1 if (drifted and args.fail_on_drift) else 0


if __name__ == "__main__":
    sys.exit(main())
