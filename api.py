"""REST API for the trained model.

    uvicorn api:app --reload            # then open http://127.0.0.1:8000/docs

Endpoints:
    GET  /health          liveness + which model is loaded
    GET  /schema          the features the model expects (types, ranges, categories)
    GET  /model           training metrics and settings
    POST /predict         {"features": {...}}          -> one prediction
    POST /predict/batch   {"rows": [{...}, {...}]}     -> many predictions

Set LOG_PREDICTIONS=0 to disable prediction logging; MODEL_DIR / LOG_DIR to relocate files.
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common import load_model, log_predictions, predict_frame

logger = logging.getLogger("prediction-api")


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature name -> value")


class BatchRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., min_length=1, max_length=1000)


def create_app(model_dir=None, log_dir=None, log_enabled=None) -> FastAPI:
    if log_enabled is None:
        log_enabled = os.environ.get("LOG_PREDICTIONS", "1").lower() not in {"0", "false", "no"}
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["pipeline"], state["meta"] = load_model(model_dir)  # fails fast if not trained
        yield
        state.clear()

    app = FastAPI(
        title="ML Prediction API",
        version="1.0.0",
        description="Serves predictions from the model trained by train.py.",
        lifespan=lifespan,
    )

    def run(df: pd.DataFrame, source: str) -> list:
        meta = state["meta"]
        try:
            result = predict_frame(state["pipeline"], meta, df)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err))
        if log_enabled:
            try:
                log_predictions(df[meta["features"]], result, source, log_dir)
            except Exception:  # never fail a prediction because logging failed
                logger.exception("Could not write the prediction log")
        return json.loads(result.to_json(orient="records"))

    @app.get("/health")
    def health():
        meta = state["meta"]
        return {
            "status": "ok",
            "model": meta["model"],
            "task": meta["task"],
            "trained_at": meta["trained_at"],
        }

    @app.get("/schema")
    def schema():
        meta = state["meta"]
        return {
            "target": meta["target"],
            "task": meta["task"],
            "classes": meta["classes"],
            "positive_class": meta["positive_class"],
            "features": meta["schema"],
        }

    @app.get("/model")
    def model_info():
        return {k: v for k, v in state["meta"].items() if k not in {"schema", "baseline"}}

    @app.post("/predict")
    def predict(request: PredictRequest):
        return run(pd.DataFrame([request.features]), "api")[0]

    @app.post("/predict/batch")
    def predict_batch(request: BatchRequest):
        predictions = run(pd.DataFrame(request.rows), "api-batch")
        return {"count": len(predictions), "predictions": predictions}

    return app


app = create_app()
