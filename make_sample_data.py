"""Generate two realistic sample datasets so you can try the system on your own-style data.

    python make_sample_data.py

Creates:
    data/sample_churn.csv               classification (target: churn = Yes/No)
    data/sample_house_prices.csv        regression     (target: price)
    data/sample_trade_CAN_IND_A.csv     SYNTHETIC annual Canada<->India trade for all HS chapters
    data/sample_macro.csv               SYNTHETIC macro indicators for Canada and India

The first two contain categorical columns, missing values and an ID column, like real data does.
The trade and macro files are made-up numbers (NOT real statistics). They exist so the
all-industry scanner can be run and tested offline; replace them with real downloads from
fetch_data.py for anything that matters.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from industries import HS2_NAMES, sector_of

DATA_DIR = Path(__file__).resolve().parent / "data"


def make_churn(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "tenure_months": rng.integers(1, 72, n).astype(float),
            "monthly_charges": rng.normal(65, 22, n).clip(15, 130).round(2),
            "support_calls": rng.poisson(1.5, n).astype(float),
            "data_usage_gb": rng.gamma(3.0, 8.0, n).round(1),
            "contract": rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20]),
            "internet_service": rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.40, 0.15]),
            "payment_method": rng.choice(["card", "bank_transfer", "cash", "e-check"], n),
        }
    )
    z = (
        -0.6
        + 1.3 * (df["contract"] == "month-to-month")
        - 0.9 * (df["contract"] == "two-year")
        - 0.035 * df["tenure_months"]
        + 0.014 * (df["monthly_charges"] - 65)
        + 0.28 * df["support_calls"]
        + 0.5 * (df["internet_service"] == "fiber")
        + 0.25 * (df["payment_method"] == "e-check")
    )
    p = 1 / (1 + np.exp(-z))
    df["churn"] = np.where(rng.random(n) < p, "Yes", "No")
    for col, frac in [("monthly_charges", 0.03), ("data_usage_gb", 0.05), ("payment_method", 0.02)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


def make_houses(n=2500, seed=11):
    rng = np.random.default_rng(seed)
    hoods = {"downtown": 1.35, "suburb": 1.0, "riverside": 1.2, "outskirts": 0.8}
    df = pd.DataFrame(
        {
            "listing_id": [f"H{i:05d}" for i in range(n)],
            "sqft": rng.normal(1700, 550, n).clip(500, 4500).round(),
            "bedrooms": rng.integers(1, 6, n).astype(float),
            "bathrooms": rng.choice([1, 1.5, 2, 2.5, 3], n),
            "age_years": rng.integers(0, 80, n).astype(float),
            "garage": rng.choice(["none", "single", "double"], n, p=[0.2, 0.4, 0.4]),
            "neighborhood": rng.choice(list(hoods), n),
        }
    )
    base = (
        60_000
        + 135 * df["sqft"]
        + 9_000 * df["bedrooms"]
        + 14_000 * df["bathrooms"]
        - 600 * df["age_years"]
        + df["garage"].map({"none": 0, "single": 9_000, "double": 18_000})
    )
    df["price"] = (base * df["neighborhood"].map(hoods) + rng.normal(0, 22_000, n)).round(-2)
    for col, frac in [("sqft", 0.02), ("age_years", 0.04), ("garage", 0.03)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


def make_trade_and_macro(seed=21, start=2000, end=2024):
    """Synthetic annual trade for every HS chapter, both directions, plus matching macro series.

    Built so there is real structure to learn: growth persists a little year to year, industries
    in the same sector move together (a shared sector factor), and a global shock (2009, 2020)
    hits everything. Noise is large, as in real trade data.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(start, end + 1)
    n = len(years)

    gdp = {"CAN": 2.2 + rng.normal(0, 0.7, n), "IND": 6.0 + rng.normal(0, 1.0, n)}
    for i, y in enumerate(years):
        shock = {2009: (-5.0, -3.0), 2020: (-6.5, -11.0), 2021: (6.0, 12.0)}.get(int(y))
        if shock:
            gdp["CAN"][i] += shock[0]
            gdp["IND"][i] += shock[1]
    glob = gdp["CAN"] + gdp["IND"]
    glob = (glob - glob.mean()) / glob.std()  # standardised global factor

    sectors = sorted({sector_of(c) for c in HS2_NAMES})
    factor = {}
    for s in sectors:
        f = np.zeros(n)
        for t in range(1, n):
            f[t] = 0.5 * f[t - 1] + rng.normal(0, 0.12)
        factor[s] = f

    rows = []
    for flow in ("X", "M"):
        for code in HS2_NAMES:
            mu = rng.normal(0.05, 0.05)
            x = rng.normal(1.0, 1.8)  # log of $ millions
            g_prev = mu
            for t, y in enumerate(years):
                g = mu + 0.35 * (g_prev - mu) + factor[sector_of(code)][t] + 0.07 * glob[t] + rng.normal(0, 0.22)
                x += g
                g_prev = g
                value = float(np.exp(x)) * 1e6
                if value < 4e5 and rng.random() < 0.35:
                    value = 0.0  # intermittent trade in small industries
                rows.append(
                    {"period": int(y), "reporter": "CAN", "partner": "IND", "flow": flow,
                     "hs2": code, "value_usd": round(value)}
                )
    trade = pd.DataFrame(rows)

    macro_rows = []
    for c, fx0, fx1, tr in (("CAN", 1.15, 1.35, 62.0), ("IND", 45.0, 83.0, 38.0)):
        fx = np.linspace(fx0, fx1, n) + rng.normal(0, 0.01 * fx1, n)
        for i, y in enumerate(years):
            macro_rows += [
                {"year": int(y), "country": c, "indicator": "gdp_growth_pct", "value": float(gdp[c][i])},
                {"year": int(y), "country": c, "indicator": "inflation_pct", "value": float(2 + rng.normal(0, 1.2))},
                {"year": int(y), "country": c, "indicator": "fx_lcu_per_usd", "value": float(fx[i])},
                {"year": int(y), "country": c, "indicator": "trade_pct_gdp", "value": float(tr + rng.normal(0, 3))},
            ]
    return trade, pd.DataFrame(macro_rows)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    make_churn().to_csv(DATA_DIR / "sample_churn.csv", index=False)
    make_houses().to_csv(DATA_DIR / "sample_house_prices.csv", index=False)
    trade, macro = make_trade_and_macro()
    trade.to_csv(DATA_DIR / "sample_trade_CAN_IND_A.csv", index=False)
    macro.to_csv(DATA_DIR / "sample_macro.csv", index=False)
    print(f"Wrote 4 sample files to {DATA_DIR}")


if __name__ == "__main__":
    main()
