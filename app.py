"""Web app for the trained model.

    streamlit run app.py
"""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from common import default_model_dir, load_model, log_predictions, predict_frame, read_logs
from monitor import compute_drift

MIN_MONITOR_ROWS = 30

st.set_page_config(page_title="ML Prediction System", layout="wide")


@st.cache_resource
def get_model(model_dir: str):
    return load_model(model_dir)


model_dir = default_model_dir()
try:
    pipeline, meta = get_model(str(model_dir))
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

schema = meta["schema"]

with st.sidebar:
    st.header("Model")
    st.markdown(f"**{meta['model']}** ({meta['task']})")
    st.caption(f"Predicts `{meta['target']}`. Trained {meta['trained_at']} on {meta['n_train']} rows.")
    log_enabled = st.checkbox("Log predictions for monitoring", value=True)
    if st.button("Reload model"):
        st.cache_resource.clear()
        st.rerun()


def show_result(row: pd.Series) -> None:
    if meta["task"] == "regression":
        st.metric(f"Predicted {meta['target']}", f"{float(row['prediction']):,.2f}")
        return
    st.metric(f"Predicted {meta['target']}", str(row["prediction"]))
    if "probability" in row:
        p = float(row["probability"])
        st.progress(min(max(p, 0.0), 1.0), text=f"P({meta['target']} = {meta['positive_class']}) = {p:.1%}")
    elif "confidence" in row:
        st.caption(f"Confidence: {float(row['confidence']):.1%}")


def score(df: pd.DataFrame, source: str) -> pd.DataFrame:
    result = predict_frame(pipeline, meta, df)
    if log_enabled:
        log_predictions(df[meta["features"]], result, source)
    return result


st.title("ML Prediction System")
tab_single, tab_batch, tab_report, tab_monitor, tab_scan = st.tabs(
    ["Single prediction", "Batch (CSV)", "Model report", "Monitoring", "Industry scanner"]
)

with tab_single:
    st.write("Fill in the values and press Predict. Defaults are the typical (median / most common) values.")
    with st.form("single"):
        columns = st.columns(3)
        values = {}
        for i, (name, spec) in enumerate(schema.items()):
            with columns[i % 3]:
                if spec["type"] == "numeric":
                    values[name] = st.number_input(
                        name,
                        value=float(spec["median"]),
                        format="%g",
                        help=f"Seen in training: {spec['min']:.4g} to {spec['max']:.4g}",
                    )
                else:
                    values[name] = st.selectbox(name, spec["choices"] or [""])
        submitted = st.form_submit_button("Predict")
    if submitted:
        show_result(score(pd.DataFrame([values]), "streamlit").iloc[0])

with tab_batch:
    st.write(f"Upload a CSV with these columns: `{', '.join(meta['features'])}`")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        data = pd.read_csv(upload)
        try:
            scored = pd.concat([data, score(data, "streamlit-batch")], axis=1)
        except ValueError as err:
            st.error(str(err))
        else:
            st.success(f"Scored {len(scored)} rows")
            st.dataframe(scored, use_container_width=True)
            st.download_button(
                "Download predictions", scored.to_csv(index=False).encode(), "predictions.csv", "text/csv"
            )

with tab_report:
    st.subheader("Held-out test performance")
    metric_cols = st.columns(max(len(meta["test_metrics"]), 1))
    for col, (name, value) in zip(metric_cols, meta["test_metrics"].items()):
        col.metric(name, value)
    st.caption(
        f"Cross-validated {meta['cv_score_label']}: {meta['cv_score']} "
        f"(a do-nothing baseline scores {meta['baseline_cv_score']})."
    )
    reports = Path(model_dir) / "reports"
    if (reports / "model_comparison.csv").exists():
        st.subheader("Models compared")
        st.dataframe(pd.read_csv(reports / "model_comparison.csv"), use_container_width=True)
    if (reports / "feature_importance.csv").exists():
        st.subheader("Which features matter most")
        importance = pd.read_csv(reports / "feature_importance.csv").head(15)
        st.bar_chart(importance.set_index("feature")["importance_mean"])
    for image in ("confusion_matrix.png", "predicted_vs_actual.png"):
        if (reports / image).exists():
            st.image(str(reports / image), width=420)

with tab_monitor:
    inputs, outputs = read_logs()
    if inputs.empty:
        st.info("No predictions logged yet. Make some predictions and come back.")
    else:
        st.write(f"{len(inputs)} logged predictions")
        if len(inputs) < MIN_MONITOR_ROWS:
            st.warning(f"Drift scores are unreliable below {MIN_MONITOR_ROWS} rows; keep scoring data.")
        features, prediction = compute_drift(meta, inputs, outputs)
        st.dataframe(features, use_container_width=True)
        if prediction and prediction["psi"] is not None:
            st.write(f"Prediction drift: PSI = {prediction['psi']:.3f} ({prediction['status']})")
        st.caption("PSI < 0.10 ok | 0.10-0.25 watch | > 0.25 drift")

with tab_scan:
    scan_reports = Path(os.environ.get("SCAN_MODEL_DIR", Path(__file__).resolve().parent / "model_scan")) / "reports"
    if not (scan_reports / "industry_scan.csv").exists():
        st.info(
            "No scan yet. Run it once, then reload this page:\n\n"
            "`python scan.py --trade data/sample_trade_CAN_IND_A.csv --macro data/sample_macro.csv`\n\n"
            "(the sample files are synthetic; use `fetch_data.py` for real Canada-India data)"
        )
    else:
        scan = pd.read_csv(scan_reports / "industry_scan.csv", dtype={"hs2": str})
        backtest = json.loads((scan_reports / "backtest.json").read_text())
        st.subheader("Every industry, ranked")
        if backtest["model_beats_best_naive_rmse"]:
            st.success(
                f"On future years it never saw, the model beat the best naive forecast by "
                f"{backtest['rmse_improvement_vs_best_naive_pct']:+.1f}% RMSE."
            )
        else:
            st.warning(
                f"The model did NOT beat the best naive forecast on held-out years "
                f"({backtest['rmse_improvement_vs_best_naive_pct']:+.1f}% RMSE). Treat the ranking with caution."
            )
        st.caption(
            f"Average rank correlation on held-out years: {backtest['rank_correlation_mean']} "
            f"(0 = no skill, 1 = perfect). Top-vs-bottom quintile spread in actual growth: "
            f"{backtest['top_minus_bottom_quintile_growth']}. Test periods: {backtest['n_test_periods']}."
        )
        left, mid, right = st.columns(3)
        flow = left.selectbox("Flow", ["all", "exports", "imports"])
        sector = mid.selectbox("Sector", ["all"] + sorted(scan["sector"].unique()))
        only_critical = right.checkbox("Critical-mineral chapters only")
        view = scan
        if flow != "all":
            view = view[view["flow"] == flow]
        if sector != "all":
            view = view[view["sector"] == sector]
        if only_critical:
            view = view[view["critical_mineral"] == 1]
        cols = ["rank", "flow", "industry", "sector", "trailing_value_usd", "last_year_growth",
                "predicted_change_pct", "signal", "reliability", "unusual_move"]
        st.dataframe(view[cols], use_container_width=True, hide_index=True)
        st.download_button("Download full scan", scan.to_csv(index=False).encode(), "industry_scan.csv", "text/csv")
        if (scan_reports / "sector_summary.csv").exists():
            st.subheader("Sectors (value-weighted)")
            st.dataframe(pd.read_csv(scan_reports / "sector_summary.csv"), use_container_width=True, hide_index=True)
        st.caption(
            "signal = predicted growth divided by the model's backtest error for that industry. "
            "The critical-mineral flag marks whole HS chapters that contain critical-mineral value "
            "chains; it is a screen, not a mineral-level answer. This is a screening aid, not investment advice."
        )
