# Machine Learning Prediction System

An end-to-end prediction system built on scikit-learn: train on your own CSV, compare and tune
models, serve predictions through a REST API or a web app, and monitor for data drift.
Classification and regression are both supported and detected automatically.

## Quick start

```bash
sudo apt install -y python3-venv python3-pip     # Ubuntu/Debian, once
python3 -m venv .venv
source .venv/bin/activate                        # prompt shows (.venv); `python` now works
pip install -r requirements.txt

python train.py                                  # demo model (breast-cancer dataset)
python predict.py --sample                       # score 5 held-out rows
streamlit run app.py                             # web app    -> http://localhost:8501
uvicorn api:app --reload                         # REST API   -> http://127.0.0.1:8000/docs
```

## Project layout

```
train.py               train, tune, evaluate, save model + reports
predict.py             command-line predictions
api.py                 FastAPI service
app.py                 Streamlit web app
monitor.py             data-drift report
common.py              shared code (schema, validation, prediction, logging)
scan.py                scan ALL industries: panel -> one model -> backtest -> ranking
build_panel.py         turns raw trade data into one cross-industry modelling table
industries.py          the 96 industries (HS chapters), sectors, critical-mineral flags
fetch_data.py          download / import real trade + macro data
make_sample_data.py    creates the sample CSVs (churn, houses, SYNTHETIC trade + macro)
data/                  sample_churn.csv (classification), sample_house_prices.csv (regression)
tests/                 pytest suite
model/                 created by train.py: model.joblib, metadata.json, reports/
logs/                  created at prediction time: predictions.jsonl
```

## 1. Use your own data

```bash
python train.py --csv mydata.csv --target <target_column> --drop <id_column>
```

- **Task detection**: a text/boolean target, or an integer target with 10 or fewer distinct values, is classification; anything else is regression. Override with `--task classification|regression`.
- **Messy data is handled**: missing values are imputed, text columns are one-hot encoded, numeric columns are scaled, and unseen categories at prediction time are ignored safely.
- **Drop ID columns** with `--drop` (comma-separated). Training warns if a column looks like an ID.
- Try the bundled samples first:
  ```bash
  python train.py --csv data/sample_churn.csv --target churn --drop customer_id
  python train.py --csv data/sample_house_prices.csv --target price --drop listing_id
  ```
- **Forecasting data (anything with dates):** add `--time-col year --gap 1` so the split is chronological (train on the past, test on the future) instead of random. A random split lets the model learn from the future and overstates accuracy.
- Training always overwrites `model/`. Use `--model-dir other_folder` to keep several models, then set `MODEL_DIR=other_folder` for the API, app and monitor.

## 2. Tuning and evaluation

`train.py` tunes three model families with grid search and cross-validation (logistic/ridge regression, random forest, histogram gradient boosting), selects the best, and checks it once on a held-out test set. Add `--quick` to skip the grid search.

It also trains a do-nothing **dummy baseline**, so you can see whether the model learned anything. Saved in `model/reports/`:

| File | Contents |
|---|---|
| `model_comparison.csv` | score and best settings for every model, plus the dummy baseline |
| `feature_importance.csv/.png` | permutation importance: how much the score drops when a feature is shuffled |
| `confusion_matrix.png` / `predicted_vs_actual.png` | test-set performance plot |
| `classification_report.txt` | precision/recall per class (classification) |
| `holdout_sample.csv` | up to 200 held-out rows, used by `predict.py --sample` |

## 3. REST API

```bash
uvicorn api:app --reload
```

Interactive docs are at `/docs`. Endpoints: `GET /health`, `GET /schema`, `GET /model`, `POST /predict`, `POST /predict/batch` (up to 1000 rows).

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"tenure_months": 5, "monthly_charges": 80, "support_calls": 3,
       "data_usage_gb": 20, "contract": "month-to-month",
       "internet_service": "fiber", "payment_method": "e-check"}}'
```

Missing columns return HTTP 422 with a clear message. Missing values and unseen categories are tolerated.

## 4. Web app

```bash
streamlit run app.py
```

Four tabs: a form to score one record (inputs are built from your model's schema), CSV upload with a downloadable results file, a model report (metrics, model comparison, feature importance), and a monitoring view.

## 5. Monitoring and tests

Every prediction from `predict.py`, the API and the web app is appended to `logs/predictions.jsonl` (disable with `--no-log` or `LOG_PREDICTIONS=0`).

```bash
python monitor.py                 # report
python monitor.py --fail-on-drift # exit code 1 on drift, for cron or CI
```

It compares logged inputs and predictions with the training distribution using the Population Stability Index: below 0.10 is ok, 0.10 to 0.25 is watch, above 0.25 is drift. PSI is noisy on small samples, so a report needs at least 30 logged rows (`--min-rows`), and even then expect occasional "watch" flags. A sustained "drift" on a feature means the model is seeing data unlike its training set, which is the cue to investigate and retrain.

Tests:

```bash
pip install -r requirements-dev.txt
pytest
```

The tests train small throwaway models in temp folders, so your real `model/` is never touched.

## 6. Scan all industries (Canada <-> India trade)

One model covers every industry at once instead of one model per industry:

```bash
python make_sample_data.py     # SYNTHETIC trade + macro, to try it offline
python scan.py --trade data/sample_trade_CAN_IND_A.csv --macro data/sample_macro.csv
```

How it works:

- **An industry is an HS chapter** (96 of them, from live animals to aircraft), split into exports and imports, so there are up to 192 series. Each is a row per year in one shared table (`build_panel.py`), described by its recent growth, volatility, share of total trade, rank, how its sector and the whole market are moving, and Canadian and Indian macro indicators.
- **The target** is next year's growth in trade value. One pooled model learns patterns that hold across industries, then scores every one.
- **No peeking at the future.** Training uses earlier years and testing uses later years, with a one-year gap so no training target overlaps the test period. Tests prove features at year *t* use only data up to *t* (and fail if you plant a leak).
- **An honest backtest.** The model is compared with naive forecasts (zero growth, repeat last year, 3-year average, whole-market growth) on years it never saw. The report says plainly whether it beats them, and how well it *ranks* industries, which is what a scanner needs.
- **Outputs** go to `model_scan/reports/`: `industry_scan.csv` (every industry ranked, with a backtest-based reliability measure and an unusual-move flag), `sector_summary.csv`, `critical_minerals.csv`, `backtest.json`. The web app's **Industry scanner** tab shows them with filters.
- **Critical minerals** are flagged at chapter level (ores, copper, nickel, aluminium, zinc, tin, other base metals, some chemicals and industrial minerals). That is a *screen*: a flagged chapter also contains ordinary goods, and individual minerals live at a finer level. See the caveats at the top of `industries.py`.

### Real data

```bash
python fetch_data.py comtrade --check --key YOUR_KEY   # one request: confirm the format first
python fetch_data.py comtrade --key YOUR_KEY           # UN Comtrade, annual, all chapters
python fetch_data.py worldbank                         # GDP growth, inflation, FX, trade % GDP
python scan.py --trade data/raw/trade_CAN_IND_A.csv --macro data/raw/macro.csv
```

- UN Comtrade needs a **free key** (https://comtradedeveloper.un.org) and rate-limits requests; the fetcher retries.
- Already downloaded a file (India's Trade Intelligence portal, Statistics Canada, ...)? Standardise it:
  `python fetch_data.py import-csv FILE --period-col Year --industry-col "HS Code" --flow-col Flow --value-col Value --scale 1000000`
- Monthly data works too (`--freq M`): the scanner detects it and adds a recent-momentum feature.
- `python fetch_data.py market` downloads FX, commodity, index and sector-ETF prices (needs `pip install yfinance`). These are saved but **not yet used** by the scanner.
- **The network commands were written from the providers' documentation and could not be run where this was built (no internet access).** The parsing is tested on provider-shaped records, but run `comtrade --check` first. If a format changed, you will see the raw response.

### Reading the results

Official trade statistics are annual or monthly and arrive weeks to months late; they are not real time. The sample data is synthetic, so its results show the pipeline works, not anything about the real economy. On real data, check `backtest.json` first: if the model does not beat the naive forecasts, the ranking is not trustworthy. It is a screening aid for where to look, not a forecast to trade on or investment advice.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `./model` | which model the API, app, predict and monitor use |
| `LOG_DIR` | `./logs` | where predictions are logged |
| `LOG_PREDICTIONS` | `1` | set to `0` to turn off API logging |
| `SCAN_MODEL_DIR` | `./model_scan` | where the web app looks for scan results |

## Notes and limits

- The saved model is tied to the scikit-learn version that trained it. Retrain after upgrading scikit-learn.
- Binary classification returns `probability` = P(positive class); the positive class is the second label in `model/metadata.json`.
- There is no authentication on the API. Put it behind a gateway before exposing it beyond your machine.
