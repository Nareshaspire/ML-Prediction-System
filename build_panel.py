"""Turn raw bilateral trade (and optional macro) data into a cross-industry modelling panel.

    python build_panel.py --trade data/raw/trade_CAN_IND_A.csv --macro data/raw/macro.csv

One row = one industry-flow series (an HS chapter x exports/imports) at one period. Every
industry is in the same table, so a single model can learn patterns that hold across industries
and then score all of them.

Target (what the model predicts)
    target_growth = log growth of the next 12 months' trade value versus the last 12 months,
    on a log1p($ millions) scale, clipped to +/-3. Using log1p($M) means micro-flows barely move
    the scale, so noise from tiny trade values is damped rather than exploding.

No look-ahead: every feature at period t uses data from t and earlier only (tests enforce this).

Rows below --min-value of trailing-12-month trade are dropped: growth on a tiny base is noise.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from industries import HS2_NAMES, reference_table

DATA_DIR = Path(__file__).resolve().parent / "data"
TARGET = "target_growth"
CATEGORICAL = ["flow", "hs2", "sector"]
BASE_NUMERIC = [
    "log_value", "g1", "g2", "g3", "g_mean", "g_vol", "zero_share", "share", "share_chg",
    "rank_pct", "market_g1", "rel_g1", "sector_g1", "critical_mineral",
]
ID_COLUMNS = ["series_id", "industry", "reporter", "partner", "period", "year", "trailing_value_usd"]


@dataclass
class Panel:
    train: pd.DataFrame   # rows whose future is known (have a target)
    latest: pd.DataFrame  # most recent period, target unknown: what the scanner scores
    features: list        # model input columns
    ppy: int              # periods per year (1 annual, 12 monthly)


def load_trade(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"hs2": str})
    missing = {"period", "flow", "hs2", "value_usd"} - set(df.columns)
    if missing:
        raise ValueError(f"Trade file is missing columns: {sorted(missing)}")
    for col in ("reporter", "partner"):
        if col not in df.columns:
            df[col] = "?"
    pairs = df[["reporter", "partner"]].drop_duplicates()
    if len(pairs) > 1:
        raise ValueError(
            "The trade file mixes several reporter/partner pairs; build one panel per pair:\n"
            + pairs.to_string(index=False)
        )
    return df


def detect_frequency(periods: pd.Series) -> int:
    return 12 if periods.max() > 9999 else 1


def _full_range(lo: int, hi: int, ppy: int) -> list:
    if ppy == 1:
        return list(range(int(lo), int(hi) + 1))
    rng = pd.period_range(f"{lo // 100}-{lo % 100:02d}", f"{hi // 100}-{hi % 100:02d}", freq="M")
    return [p.year * 100 + p.month for p in rng]


def _macro_wide(macro: pd.DataFrame, last_year: int) -> pd.DataFrame:
    wide = macro.pivot_table(index="year", columns=["country", "indicator"], values="value", aggfunc="mean")
    wide.columns = [f"macro_{c}_{i}" for c, i in wide.columns]
    wide = wide.reindex(range(int(wide.index.min()), last_year + 1)).ffill()  # latest year often unpublished
    return wide


def build_panel(trade: pd.DataFrame, macro: pd.DataFrame | None = None, freq: str = "auto",
                min_value: float = 1e6) -> Panel:
    trade = trade.copy()
    trade["hs2"] = trade["hs2"].astype(str).str.zfill(2)
    trade = trade[trade["hs2"].isin(HS2_NAMES)]
    trade["series_id"] = trade["flow"].astype(str) + "_" + trade["hs2"]
    reporter, partner = trade["reporter"].iloc[0], trade["partner"].iloc[0]

    ppy = {"A": 1, "M": 12}.get(freq) or detect_frequency(trade["period"])
    V = trade.pivot_table(index="period", columns="series_id", values="value_usd", aggfunc="sum")
    V = V.reindex(_full_range(V.index.min(), V.index.max(), ppy)).fillna(0.0)  # no record = no trade
    if len(V) < 4 * ppy + 1:
        raise ValueError("Need at least 4 years of history to build features and a target.")
    cols = list(V.columns)

    S = V.rolling(ppy, min_periods=ppy).sum()  # trailing-12-month value (= the value itself if annual)
    L = np.log1p(S / 1e6)
    g1 = L - L.shift(ppy)
    g2 = L.shift(ppy) - L.shift(2 * ppy)
    g3 = L.shift(2 * ppy) - L.shift(3 * ppy)
    stack = np.stack([g1.to_numpy(), g2.to_numpy(), g3.to_numpy()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices are expected early on
        g_mean = pd.DataFrame(np.nanmean(stack, axis=0), index=V.index, columns=cols)
        g_vol = pd.DataFrame(np.nanstd(stack, axis=0, ddof=1), index=V.index, columns=cols)
    zero_share = (V == 0).astype(float).rolling(3 * ppy, min_periods=1).mean()

    share = pd.DataFrame(0.0, index=V.index, columns=cols)
    rank_pct = pd.DataFrame(np.nan, index=V.index, columns=cols)
    market_g1 = pd.DataFrame(np.nan, index=V.index, columns=cols)
    for flow in sorted({c.split("_")[0] for c in cols}):
        fc = [c for c in cols if c.startswith(flow + "_")]
        total = S[fc].sum(axis=1)
        share[fc] = S[fc].div(total.replace(0, np.nan), axis=0)
        rank_pct[fc] = S[fc].rank(axis=1, pct=True)
        ml = np.log1p(total / 1e6)
        mg = ml - ml.shift(ppy)
        for c in fc:
            market_g1[c] = mg
    frames = {
        "trailing_value_usd": S, "log_value": L, "g1": g1, "g2": g2, "g3": g3, "g_mean": g_mean,
        "g_vol": g_vol, "zero_share": zero_share, "share": share, "share_chg": share - share.shift(ppy),
        "rank_pct": rank_pct, "market_g1": market_g1, TARGET: (L.shift(-ppy) - L).clip(-3, 3),
    }
    extra_numeric = []
    if ppy > 1:
        L3 = np.log1p(V.rolling(3, min_periods=3).sum() / 1e6)
        frames["mom_recent"] = L3 - L3.shift(ppy)  # last 3 months vs the same 3 months a year ago
        extra_numeric.append("mom_recent")

    index = pd.MultiIndex.from_product([V.index, V.columns], names=["period", "series_id"])
    df = pd.DataFrame({k: f.to_numpy().ravel() for k, f in frames.items()}, index=index).reset_index()

    df["flow"] = df["series_id"].str[0]
    df["hs2"] = df["series_id"].str[2:]
    ref = reference_table().set_index("hs2")
    df["industry"] = df["hs2"].map(ref["industry"])
    df["sector"] = df["hs2"].map(ref["sector"])
    df["critical_mineral"] = df["hs2"].map(ref["critical_mineral"]).astype(int)
    df["rel_g1"] = df["g1"] - df["market_g1"]
    df["sector_g1"] = df.groupby(["period", "flow", "sector"])["g1"].transform("mean")
    df["year"] = df["period"] if ppy == 1 else df["period"] // 100
    if ppy > 1:
        df["month"] = (df["period"] % 100).astype(str).str.zfill(2)
    df["reporter"], df["partner"] = reporter, partner

    macro_cols = []
    if macro is not None and len(macro):
        wide = _macro_wide(macro, int(df["year"].max()))
        df["_macro_year"] = df["year"] if ppy == 1 else df["year"] - 1  # monthly rows use the last full year
        df = df.merge(wide, left_on="_macro_year", right_index=True, how="left").drop(columns="_macro_year")
        macro_cols = list(wide.columns)

    features = CATEGORICAL + BASE_NUMERIC + extra_numeric + (["month"] if ppy > 1 else []) + macro_cols
    # Keep rows with a material trade base and at least one year of history (g1 defined).
    df = df[(df["trailing_value_usd"] >= min_value) & df["g1"].notna()]
    df = df.sort_values(["period", "series_id"]).reset_index(drop=True)

    last = df["period"].max()
    latest = df[df["period"] == last].reset_index(drop=True)
    train = df[df[TARGET].notna()].reset_index(drop=True)
    return Panel(train=train, latest=latest, features=features, ppy=ppy)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trade", required=True, help="Trade CSV from fetch_data.py (or the sample file)")
    parser.add_argument("--macro", help="Optional macro CSV from fetch_data.py")
    parser.add_argument("--freq", choices=["auto", "A", "M"], default="auto")
    parser.add_argument("--min-value", type=float, default=1e6, help="Min trailing-12m trade value in USD")
    parser.add_argument("--out", default=str(DATA_DIR / "panel.csv"))
    args = parser.parse_args()

    macro = pd.read_csv(args.macro) if args.macro else None
    panel = build_panel(load_trade(args.trade), macro, args.freq, args.min_value)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.train.to_csv(out, index=False)
    latest_path = out.with_name(out.stem + "_latest.csv")
    panel.latest.to_csv(latest_path, index=False)
    print(f"{len(panel.train)} training rows, {panel.train['series_id'].nunique()} industry-flow series -> {out}")
    print(f"{len(panel.latest)} rows for the latest period -> {latest_path}")


if __name__ == "__main__":
    main()
