"""Tests for training, prediction, the API, the web app and monitoring.

Run with:  pytest
Every test trains on small synthetic data in a temp folder, so your real model is untouched.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common import (
    build_schema,
    load_model,
    log_predictions,
    predict_frame,
    prepare_input,
    read_logs,
)
from monitor import compute_drift, psi
from train import detect_task, train

ROOT = Path(__file__).resolve().parent.parent


def make_classification(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "income": rng.normal(50_000, 12_000, n),
            "visits": rng.integers(0, 20, n).astype(float),
            "plan": rng.choice(["basic", "pro", "team"], n),
        }
    )
    logit = 0.08 * (X["age"] - 40) + 0.00008 * (X["income"] - 50_000) + (X["plan"] == "pro") - 0.1 * X["visits"] + 1
    y = pd.Series(np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no"), name="label")
    X.loc[::17, "income"] = np.nan  # missing numeric values
    X.loc[::23, "plan"] = np.nan  # missing categorical values
    return X, y


def make_regression(n=400, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"sqft": rng.normal(1500, 400, n), "rooms": rng.integers(1, 7, n).astype(float),
                      "area": rng.choice(["north", "south"], n)})
    y = pd.Series(100 * X["sqft"] + 5000 * X["rooms"] + rng.normal(0, 5000, n), name="price")
    return X, y


@pytest.fixture(scope="module")
def clf(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clf") / "model"
    X, y = make_classification()
    meta = train(X, y, model_dir=folder, quick=True, target_name="label", verbose=False)
    return folder, X, y, meta


# ----------------------------------------------------------------------------- training
def test_detect_task():
    assert detect_task(pd.Series(["a", "b", "a"])) == "classification"
    assert detect_task(pd.Series([0, 1, 1, 0])) == "classification"
    assert detect_task(pd.Series(np.linspace(0, 100, 50))) == "regression"
    assert detect_task(pd.Series([1.5, 2.5, 3.5]), "classification") == "classification"


def test_classification_training_creates_artifacts(clf):
    folder, _, _, meta = clf
    assert meta["task"] == "classification"
    assert meta["positive_class"] == "yes"
    assert meta["test_metrics"]["roc_auc"] > 0.6  # clearly better than the 0.5 dummy
    for name in ["model.joblib", "metadata.json", "reports/model_comparison.csv",
                 "reports/feature_importance.csv", "reports/holdout_sample.csv"]:
        assert (folder / name).exists(), name


def test_regression_training(tmp_path):
    X, y = make_regression()
    meta = train(X, y, model_dir=tmp_path / "m", quick=True, target_name="price", verbose=False)
    assert meta["task"] == "regression"
    assert meta["test_metrics"]["r2"] > 0.9
    assert meta["cv_score"] < meta["baseline_cv_score"]  # RMSE beats the dummy


def test_training_rejects_tiny_or_single_class_data(tmp_path):
    X, y = make_classification(n=10)
    with pytest.raises(ValueError):
        train(X, y, model_dir=tmp_path, quick=True, verbose=False)
    X, _ = make_classification()
    with pytest.raises(ValueError):
        train(X, pd.Series(["same"] * len(X)), model_dir=tmp_path, quick=True, verbose=False)


# ----------------------------------------------------------------------------- prediction
def test_predict_handles_missing_and_unseen_values(clf):
    folder, X, _, _ = clf
    pipeline, meta = load_model(folder)
    rows = X.head(3).copy()
    rows.loc[rows.index[0], "income"] = None
    rows.loc[rows.index[1], "plan"] = "brand-new-plan"  # never seen in training
    result = predict_frame(pipeline, meta, rows)
    assert len(result) == 3
    assert set(result["prediction"]) <= {"yes", "no"}
    assert result["probability"].between(0, 1).all()


def test_missing_column_is_a_clear_error(clf):
    folder, X, _, _ = clf
    pipeline, meta = load_model(folder)
    with pytest.raises(ValueError, match="Missing required columns"):
        predict_frame(pipeline, meta, X.drop(columns=["age"]).head(2))


def test_prepare_input_coerces_types():
    X, _ = make_classification(50)
    schema = build_schema(X)
    messy = pd.DataFrame({"age": ["41.5"], "income": ["oops"], "visits": [3], "plan": [None]})
    clean = prepare_input(messy, schema)
    assert clean.loc[0, "age"] == 41.5
    assert np.isnan(clean.loc[0, "income"]) and pd.isna(clean.loc[0, "plan"])


# ----------------------------------------------------------------------------- monitoring
def test_psi_basics():
    same = [0.25, 0.25, 0.25, 0.25]
    assert psi(same, same) == pytest.approx(0.0, abs=1e-9)
    assert psi(same, [0.7, 0.1, 0.1, 0.1]) > 0.25


def test_log_roundtrip(clf, tmp_path):
    folder, X, _, _ = clf
    pipeline, meta = load_model(folder)
    rows = X.head(5)
    result = predict_frame(pipeline, meta, rows)
    assert log_predictions(rows, result, "test", log_dir=tmp_path) == 5
    inputs, outputs = read_logs(tmp_path)
    assert len(inputs) == 5 and list(outputs["source"].unique()) == ["test"]
    assert read_logs(tmp_path / "nothing-here")[0].empty


def test_drift_detects_a_shifted_feature(clf):
    folder, X, _, _ = clf
    _, meta = load_model(folder)
    stable, _ = compute_drift(meta, X, pd.DataFrame())
    assert (stable["status"] == "ok").all()

    shifted = X.copy()
    shifted["age"] = shifted["age"] + 30  # customers suddenly 30 years older
    features, _ = compute_drift(meta, shifted, pd.DataFrame())
    by_name = features.set_index("feature")["status"]
    assert by_name["age"] == "drift"
    assert by_name["visits"] == "ok"


def test_drift_detects_unseen_categories(clf):
    folder, X, _, _ = clf
    _, meta = load_model(folder)
    changed = X.copy()
    changed["plan"] = "enterprise"
    features, _ = compute_drift(meta, changed, pd.DataFrame())
    assert features.set_index("feature").loc["plan", "status"] == "drift"


# ----------------------------------------------------------------------------- API
def test_api_end_to_end(clf, tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api import create_app

    folder, X, _, _ = clf
    row = X.iloc[0].where(X.iloc[0].notna(), None).to_dict()
    with TestClient(create_app(model_dir=folder, log_dir=tmp_path)) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert set(client.get("/schema").json()["features"]) == set(X.columns)
        assert client.get("/model").json()["task"] == "classification"

        one = client.post("/predict", json={"features": row})
        assert one.status_code == 200 and one.json()["prediction"] in {"yes", "no"}

        batch = client.post("/predict/batch", json={"rows": [row, row, row]})
        assert batch.status_code == 200 and batch.json()["count"] == 3

        bad = client.post("/predict", json={"features": {"age": 30}})
        assert bad.status_code == 422 and "Missing required columns" in bad.json()["detail"]
        assert client.post("/predict/batch", json={"rows": []}).status_code == 422

    inputs, outputs = read_logs(tmp_path)
    assert len(inputs) == 4  # 1 single + 3 batch; the rejected request is not logged


def test_api_refuses_to_start_without_a_model(tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api import create_app

    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(model_dir=tmp_path / "empty")):
            pass


# ----------------------------------------------------------------------------- web app
def test_streamlit_app_runs_and_predicts(clf, tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    folder, _, _, _ = clf
    monkeypatch.setenv("MODEL_DIR", str(folder))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    predict_button = [b for b in at.button if b.label == "Predict"][0]
    at = predict_button.click().run()
    assert not at.exception
    assert any("Predicted" in m.label for m in at.metric)
    assert (tmp_path / "logs" / "predictions.jsonl").exists()
