"""Scan ALL industries at once: train one pooled model and rank every industry's outlook.

    python scan.py --trade data/sample_trade_CAN_IND_A.csv --macro data/sample_macro.csv
    python scan.py --trade data/raw/trade_CAN_IND_A.csv --macro data/raw/macro.csv   # real data
    python scan.py --trade <new trade file> --no-train                               # re-score only

What it does
  1. Builds a panel with every HS chapter x exports/imports as its own series (build_panel.py).
  2. Trains ONE model on all of them with a chronological split (past -> future, with a gap, so
     the model never sees the period it is tested on).
  3. Backtests it honestly against naive forecasts (zero growth, repeat last year's growth, ...).
  4. Scores every industry for the next 12 months and writes ranked tables to
     <model-dir>/reports/: industry_scan.csv, sector_summary.csv, critical_minerals.csv,
     backtest.json.

Treat the output as a screening aid, not a forecast to trade on: see the backtest verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_panel import TARGET, build_panel, load_trade
from common import BASE_DIR, load_model, predict_frame
from industries import CRITICAL_MINERAL_CHAPTERS
from train import train

DEFAULT_MODEL_DIR = BASE_DIR / "model_scan"
FLOW_NAMES = {"X": "exports", "M": "imports"}


# --------------------------------------------------------------------------- backtest
def backtest(model_dir: Path, panel_train: pd.DataFrame) -> dict:
    """Compare the model with naive forecasts on the held-out (future) periods."""
    tp = pd.read_csv(model_dir / "reports" / "test_predictions.csv", dtype={"series_id": str})
    tp = tp.merge(
        panel_train[["period", "series_id", "g1", "g_mean", "market_g1"]], on=["period", "series_id"], how="left"
    )
    candidates = {
        "model": tp["predicted"],
        "naive: zero growth": pd.Series(0.0, index=tp.index),
        "naive: repeat last year's growth": tp["g1"].fillna(0.0),
        "naive: 3-year average growth": tp["g_mean"].fillna(0.0),
        "naive: whole-market growth": tp["market_g1"].fillna(0.0),
    }
    rows = []
    for name, pred in candidates.items():
        err = tp["actual"] - pred
        nonzero = (pred != 0) & (tp["actual"] != 0)
        rows.append(
            {
                "forecast": name,
                "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
                "mae": round(float(err.abs().mean()), 4),
                "direction_accuracy": round(float((np.sign(pred) == np.sign(tp["actual"]))[nonzero].mean()), 3)
                if nonzero.any() and name != "naive: zero growth" else None,
            }
        )
    table = pd.DataFrame(rows)

    # Does the model rank industries well? (what a scanner actually needs)
    ics, spreads = [], []
    for _, g in tp.groupby("period"):
        if len(g) < 10:
            continue
        ics.append(g["predicted"].corr(g["actual"], method="spearman"))
        quint = pd.qcut(g["predicted"].rank(method="first"), 5, labels=False)
        spreads.append(g.loc[quint == 4, "actual"].mean() - g.loc[quint == 0, "actual"].mean())
    naive_best = table[table["forecast"] != "model"]["rmse"].min()
    model_rmse = float(table.loc[table["forecast"] == "model", "rmse"].iloc[0])
    return {
        "table": table.to_dict(orient="records"),
        "rank_correlation_mean": round(float(np.nanmean(ics)), 3) if ics else None,
        "top_minus_bottom_quintile_growth": round(float(np.nanmean(spreads)), 3) if spreads else None,
        "n_test_periods": len(ics),
        "model_beats_best_naive_rmse": bool(model_rmse < naive_best),
        "rmse_improvement_vs_best_naive_pct": round(100 * (naive_best - model_rmse) / naive_best, 1),
    }


# --------------------------------------------------------------------------- scoring
def score_industries(model_dir: Path, panel, bt_predictions: pd.DataFrame) -> pd.DataFrame:
    pipeline, meta = load_model(model_dir)
    latest = panel.latest.copy()
    result = predict_frame(pipeline, meta, latest[meta["features"]])
    out = latest[
        ["series_id", "flow", "hs2", "industry", "sector", "critical_mineral", "period", "trailing_value_usd", "g1"]
    ].copy()
    out["flow"] = out["flow"].map(FLOW_NAMES)
    out = out.rename(columns={"g1": "last_year_growth"})
    out["predicted_growth"] = result["prediction"].to_numpy()

    base = out["trailing_value_usd"] / 1e6
    next_value = ((1 + base) * np.exp(out["predicted_growth"]) - 1) * 1e6
    out["predicted_value_usd"] = next_value.clip(lower=0)
    out["predicted_change_pct"] = 100 * (out["predicted_value_usd"] / out["trailing_value_usd"] - 1)

    # How trustworthy has the model been on this industry? Fall back to its sector, then overall.
    bt = bt_predictions.assign(abs_err=(bt_predictions["actual"] - bt_predictions["predicted"]).abs())
    by_series = bt.groupby("series_id")["abs_err"].agg(["mean", "count"])
    sector_of_series = panel.train.drop_duplicates("series_id").set_index("series_id")["sector"]
    bt = bt.assign(sector=bt["series_id"].map(sector_of_series))
    by_sector = bt.groupby("sector")["abs_err"].mean()
    overall = float(bt["abs_err"].mean())
    mae, n = [], []
    for sid, sector in zip(out["series_id"], out["sector"]):
        if sid in by_series.index and by_series.loc[sid, "count"] >= 3:
            mae.append(by_series.loc[sid, "mean"])
            n.append(int(by_series.loc[sid, "count"]))
        else:
            mae.append(by_sector.get(sector, overall))
            n.append(0)
    out["backtest_mae"] = np.round(mae, 4)
    out["backtest_obs"] = n
    out["signal"] = (out["predicted_growth"] / out["backtest_mae"]).round(2)
    out["reliability"] = np.where(out["backtest_mae"] <= overall, "above average", "below average")

    # Unusual move: this year's growth versus the industry's own history.
    hist = panel.train.groupby("series_id")["g1"].agg(["mean", "std"])
    z = (out["last_year_growth"] - out["series_id"].map(hist["mean"])) / out["series_id"].map(hist["std"])
    out["last_move_zscore"] = z.round(2)
    out["unusual_move"] = z.abs() > 2

    out["critical_mineral_note"] = out["hs2"].map(CRITICAL_MINERAL_CHAPTERS).fillna("")
    out["predicted_growth"] = out["predicted_growth"].round(4)
    out["last_year_growth"] = out["last_year_growth"].round(4)
    out["predicted_change_pct"] = out["predicted_change_pct"].round(1)
    out["predicted_value_usd"] = out["predicted_value_usd"].round(0)
    out = out.sort_values("signal", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def sector_summary(scan: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        w = g["trailing_value_usd"]
        return pd.Series(
            {
                "industries": len(g),
                "trailing_value_usd": w.sum(),
                "predicted_growth_value_weighted": float((g["predicted_growth"] * w).sum() / w.sum()),
                "predicted_change_pct": 100 * (g["predicted_value_usd"].sum() / w.sum() - 1),
                "avg_signal": g["signal"].mean(),
                "n_unusual_moves": int(g["unusual_move"].sum()),
            }
        )

    out = scan.groupby(["flow", "sector"]).apply(agg, include_groups=False).reset_index()
    return out.sort_values(["flow", "predicted_growth_value_weighted"], ascending=[True, False]).round(3)


# --------------------------------------------------------------------------- pipeline
def run_scan(trade: pd.DataFrame, macro=None, model_dir=None, freq="auto", min_value=1e6,
             quick=False, test_size=0.2, seed=42, retrain=True, verbose=True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    panel = build_panel(trade, macro, freq, min_value)
    say(
        f"Panel: {len(panel.train)} labelled rows, {panel.train['series_id'].nunique()} industry-flow series "
        f"across {panel.train['sector'].nunique()} sectors; scoring {len(panel.latest)} at the latest period.\n"
    )
    if retrain:
        train(
            panel.train[panel.features], panel.train[TARGET], task="regression", model_dir=model_dir,
            quick=quick, test_size=test_size, seed=seed, target_name=TARGET,
            data_source="industry panel (build_panel.py)", verbose=verbose,
            time_values=panel.train["period"], gap=panel.ppy,
            extra=panel.train[["series_id", "industry"]],
        )
    else:
        _, meta = load_model(model_dir)
        if set(meta["features"]) != set(panel.features):
            raise SystemExit("Saved model was trained on different features; rerun without --no-train.")

    bt = backtest(model_dir, panel.train)
    reports = model_dir / "reports"
    (reports / "backtest.json").write_text(json.dumps(bt, indent=2))
    bt_predictions = pd.read_csv(reports / "test_predictions.csv", dtype={"series_id": str})
    scan = score_industries(model_dir, panel, bt_predictions)
    sectors = sector_summary(scan)
    critical = scan[scan["critical_mineral"] == 1].sort_values("trailing_value_usd", ascending=False)

    scan.to_csv(reports / "industry_scan.csv", index=False)
    sectors.to_csv(reports / "sector_summary.csv", index=False)
    critical.to_csv(reports / "critical_minerals.csv", index=False)
    return {"scan": scan, "sectors": sectors, "critical": critical, "backtest": bt,
            "panel": panel, "model_dir": model_dir}


def print_report(result: dict) -> None:
    bt, scan = result["backtest"], result["scan"]
    print("\n" + "=" * 78)
    print("BACKTEST on future periods the model never saw (lower RMSE is better)")
    print(pd.DataFrame(bt["table"]).to_string(index=False))
    verdict = "BEATS" if bt["model_beats_best_naive_rmse"] else "does NOT beat"
    print(f"\nThe model {verdict} the best naive forecast "
          f"({bt['rmse_improvement_vs_best_naive_pct']:+.1f}% RMSE).")
    if bt["rank_correlation_mean"] is not None:
        print(f"Ranking skill: average rank correlation {bt['rank_correlation_mean']:+.2f}; "
              f"top-quintile minus bottom-quintile actual growth {bt['top_minus_bottom_quintile_growth']:+.3f} "
              f"(over {bt['n_test_periods']} test periods).")

    cols = ["rank", "flow", "industry", "sector", "trailing_value_usd", "last_year_growth",
            "predicted_change_pct", "signal", "reliability"]
    fmt = scan[cols].assign(trailing_value_usd=lambda d: (d["trailing_value_usd"] / 1e6).round(1))
    fmt = fmt.rename(columns={"trailing_value_usd": "value_$M", "last_year_growth": "last_yr_g"})
    print("\n" + "=" * 78)
    print(f"ALL {len(scan)} INDUSTRIES SCANNED - strongest outlook (signal = predicted growth / backtest error)")
    print(fmt.head(10).to_string(index=False))
    print("\nWeakest outlook")
    print(fmt.tail(10).to_string(index=False))
    print("\n" + "=" * 78)
    print("SECTORS (value-weighted)")
    print(result["sectors"][["flow", "sector", "industries", "predicted_change_pct", "avg_signal"]].to_string(index=False))
    crit = result["critical"]
    if len(crit):
        print("\n" + "=" * 78)
        print("CRITICAL-MINERAL-EXPOSED INDUSTRIES (chapter-level screen; see industries.py caveats)")
        c = crit[["flow", "industry", "trailing_value_usd", "predicted_change_pct", "signal"]].copy()
        c["trailing_value_usd"] = (c["trailing_value_usd"] / 1e6).round(1)
        print(c.rename(columns={"trailing_value_usd": "value_$M"}).to_string(index=False))
    print(f"\nFull tables saved in {result['model_dir'] / 'reports'}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trade", required=True, help="Trade CSV (fetch_data.py output or the sample file)")
    parser.add_argument("--macro", help="Optional macro CSV")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="default: ./model_scan")
    parser.add_argument("--freq", choices=["auto", "A", "M"], default="auto")
    parser.add_argument("--min-value", type=float, default=1e6, help="Min trailing-12m trade value in USD")
    parser.add_argument("--quick", action="store_true", help="Skip hyper-parameter search")
    parser.add_argument("--no-train", action="store_true", help="Re-score with the saved model, no retraining")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        result = run_scan(
            load_trade(args.trade), pd.read_csv(args.macro) if args.macro else None, args.model_dir,
            args.freq, args.min_value, args.quick, args.test_size, args.seed, retrain=not args.no_train,
        )
    except (ValueError, FileNotFoundError) as err:
        raise SystemExit(f"Error: {err}")
    print_report(result)


if __name__ == "__main__":
    main()
