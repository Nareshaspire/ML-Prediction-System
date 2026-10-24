"""Download or import the real data the all-industry scanner needs.

Outputs go to data/raw/ (override with --out-dir) in the exact formats scan.py reads:

    trade_CAN_IND_A.csv   period, reporter, partner, flow (X|M), hs2, value_usd   (A = annual, M = monthly)
    macro.csv             year, country, indicator, value
    market_monthly.csv    period + monthly closes: FX, commodities, indices, sector ETFs (optional)

Canada's perspective: flow X = Canada exports to India, M = Canada imports from India.

Commands
    python fetch_data.py comtrade --key YOUR_KEY                # UN Comtrade, annual, all 96 chapters
    python fetch_data.py comtrade --key YOUR_KEY --freq M       # monthly (more rows, more API calls)
    python fetch_data.py comtrade --check                       # one request; prints the raw response
    python fetch_data.py worldbank                              # GDP growth, inflation, FX, trade % GDP
    python fetch_data.py import-csv FILE --period-col .. --industry-col .. --flow-col .. --value-col ..
    python fetch_data.py market                                 # optional, needs `pip install yfinance`

Then:  python scan.py --trade data/raw/trade_CAN_IND_A.csv --macro data/raw/macro.csv

NOTE: the network commands (comtrade, worldbank, market) are written from the providers'
documentation and could not be run in the build environment (no internet access there). Run
`comtrade --check` first; if a provider changed its format, the error shows what came back.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from industries import MACRO_TICKERS, all_hs2, reference_table

BASE_DIR = Path(__file__).resolve().parent
COMTRADE_URL = "https://comtradeapi.un.org/data/v1/get/C/{freq}/HS"
WORLD_BANK_URL = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
CODES = {"CAN": 124, "IND": 699}  # UN Comtrade numeric country codes (ISO-3 used in the output files)
WORLD_BANK_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
    "FP.CPI.TOTL.ZG": "inflation_pct",
    "PA.NUS.FCRF": "fx_lcu_per_usd",
    "NE.TRD.GNFS.ZS": "trade_pct_gdp",
}
TRADE_COLUMNS = ["period", "reporter", "partner", "flow", "hs2", "value_usd"]


def default_out_dir() -> Path:
    return Path(os.environ.get("RAW_DATA_DIR", BASE_DIR / "data" / "raw"))


# --------------------------------------------------------------------------- shared cleaning
def finish_trade(df: pd.DataFrame, reporter: str, partner: str) -> pd.DataFrame:
    """Aggregate to one row per (period, flow, hs2); keep valid HS-2 chapters only."""
    df = df.copy()
    df["hs2"] = df["hs2"].astype(str).str.extract(r"(\d+)")[0].str.zfill(2).str[:2]
    df = df[df["hs2"].isin(all_hs2())]
    df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce")
    df = df.dropna(subset=["value_usd"])
    df = df[df["value_usd"] >= 0]
    out = df.groupby(["period", "flow", "hs2"], as_index=False)["value_usd"].sum()
    out["reporter"], out["partner"] = reporter, partner
    return out[TRADE_COLUMNS].sort_values(["flow", "hs2", "period"]).reset_index(drop=True)


def parse_period(s: pd.Series, freq: str) -> pd.Series:
    """Return ints: YYYY for annual data, YYYYMM for monthly."""
    text = s.astype(str).str.strip()
    if freq == "A":
        return pd.to_numeric(text.str.extract(r"(\d{4})")[0], errors="coerce")
    digits = text.str.fullmatch(r"\d{6}")
    out = pd.Series(index=s.index, dtype="float64")
    out[digits] = pd.to_numeric(text[digits])
    rest = ~digits
    if rest.any():
        parsed = pd.to_datetime(text[rest], errors="coerce", format="mixed")
        out[rest] = parsed.dt.year * 100 + parsed.dt.month
    return out


def write_trade(df: pd.DataFrame, out_dir: Path, freq: str, source: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trade_{df['reporter'].iloc[0]}_{df['partner'].iloc[0]}_{freq}.csv"
    df.to_csv(path, index=False)
    (out_dir / "trade_source.json").write_text(json.dumps(source, indent=2))
    print(f"Wrote {len(df)} rows, {df['hs2'].nunique()} industries, periods {df['period'].min()}-{df['period'].max()} -> {path}")
    return path


# --------------------------------------------------------------------------- UN Comtrade
def parse_comtrade_records(records: list, flow_names: dict) -> pd.DataFrame:
    """Convert the `data` list of a Comtrade v1 response into raw trade rows."""
    rows = []
    for r in records:
        value = r.get("primaryValue")
        if value is None:
            value = r.get("fobvalue") if r.get("flowCode") == "X" else r.get("cifvalue")
        rows.append(
            {
                "period": r.get("period") if r.get("period") is not None else r.get("refYear"),
                "flow": flow_names.get(str(r.get("flowCode"))),
                "hs2": r.get("cmdCode"),
                "value_usd": value,
            }
        )
    return pd.DataFrame(rows, columns=["period", "flow", "hs2", "value_usd"]).dropna(subset=["flow"])


def comtrade_get(session, url: str, params: dict, key: str, retries: int = 4) -> list:
    headers = {"Ocp-Apim-Subscription-Key": key} if key else {}
    for attempt in range(retries):
        resp = session.get(url, params=params, headers=headers, timeout=90)
        if resp.status_code == 429:  # rate limited: wait and retry
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code in (401, 403):
            raise SystemExit(
                f"Comtrade rejected the request (HTTP {resp.status_code}). A free key is required: "
                "register at https://comtradedeveloper.un.org, then pass --key or set COMTRADE_API_KEY."
            )
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict) or "data" not in body:
            raise SystemExit(f"Unexpected Comtrade response (no 'data' field): {str(body)[:300]}")
        return body["data"] or []
    raise SystemExit("Comtrade kept rate-limiting the request; wait a few minutes and run again.")


def cmd_comtrade(args):
    import requests

    key = args.key or os.environ.get("COMTRADE_API_KEY", "")
    if not key:
        raise SystemExit("Set COMTRADE_API_KEY or pass --key. Free keys: https://comtradedeveloper.un.org")
    reporter, partner = args.reporter.upper(), args.partner.upper()
    url = COMTRADE_URL.format(freq=args.freq)
    session = requests.Session()
    base = {"reporterCode": CODES[reporter], "partnerCode": CODES[partner], "cmdCode": "AG2"}
    flows = {"X": "X", "M": "M"}

    if args.check:
        period = str(args.end_year) if args.freq == "A" else f"{args.end_year}12"
        records = comtrade_get(session, url, {**base, "flowCode": "X", "period": period}, key)
        print(f"{len(records)} records for period {period}. First record:")
        print(json.dumps(records[0] if records else {}, indent=2))
        return

    if args.freq == "A":
        chunks = [[str(y)] for y in range(args.start_year, args.end_year + 1)]
        chunks = [sum(chunks[i : i + 5], []) for i in range(0, len(chunks), 5)]  # 5 years per request
    else:
        months = [m.strftime("%Y%m") for m in pd.period_range(f"{args.start_year}-01", f"{args.end_year}-12", freq="M")]
        chunks = [months[i : i + 12] for i in range(0, len(months), 12)]

    frames = []
    for chunk in chunks:
        for code in flows:
            records = comtrade_get(session, url, {**base, "flowCode": code, "period": ",".join(chunk)}, key)
            frames.append(parse_comtrade_records(records, flows))
            print(f"  {chunk[0]}..{chunk[-1]} flow {code}: {len(records)} records")
            time.sleep(1.0)  # be polite to the free tier
    raw = pd.concat(frames, ignore_index=True)
    if raw.empty:
        raise SystemExit("Comtrade returned no rows. Run with --check to see a raw response.")
    raw["period"] = parse_period(raw["period"], args.freq)
    raw = raw.dropna(subset=["period"]).astype({"period": int})
    write_trade(finish_trade(raw, reporter, partner), Path(args.out_dir), args.freq,
                {"source": "UN Comtrade", "freq": args.freq, "reporter": reporter, "partner": partner})


# --------------------------------------------------------------------------- World Bank
def cmd_worldbank(args):
    import requests

    session = requests.Session()
    rows = []
    for indicator, name in WORLD_BANK_INDICATORS.items():
        url = WORLD_BANK_URL.format(countries="CAN;IND", indicator=indicator)
        resp = session.get(url, params={"format": "json", "per_page": 2000, "date": f"{args.start_year}:{args.end_year}"},
                           timeout=60)
        resp.raise_for_status()
        body = resp.json()
        if not (isinstance(body, list) and len(body) == 2):
            raise SystemExit(f"Unexpected World Bank response for {indicator}: {str(body)[:300]}")
        for r in body[1] or []:
            if r.get("value") is not None:
                rows.append({"year": int(r["date"]), "country": r["countryiso3code"],
                             "indicator": name, "value": float(r["value"])})
        print(f"  {indicator} ({name}): {sum(1 for r in rows if r['indicator'] == name)} values")
    out = pd.DataFrame(rows, columns=["year", "country", "indicator", "value"])
    if out.empty:
        raise SystemExit("World Bank returned no data.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.sort_values(["country", "indicator", "year"]).to_csv(out_dir / "macro.csv", index=False)
    print(f"Wrote {len(out)} rows -> {out_dir / 'macro.csv'}")


# --------------------------------------------------------------------------- your own download
def cmd_import_csv(args):
    """Standardise a file you downloaded yourself (India's TIA portal, StatCan CIMT, ...)."""
    df = pd.read_csv(args.file, low_memory=False, dtype=str)
    for col in (args.period_col, args.industry_col, args.flow_col, args.value_col):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found in {args.file}. Columns are: {list(df.columns)}")
    mapping = {k.strip().lower(): v.strip().upper() for k, v in (p.split("=") for p in args.flow_map.split(","))}
    flows = df[args.flow_col].str.strip().str.lower().map(mapping)
    if flows.isna().all():
        sample = sorted(df[args.flow_col].dropna().unique())[:5]
        raise SystemExit(f"--flow-map matched no rows. Flow values in the file look like: {sample}")
    raw = pd.DataFrame(
        {
            "period": parse_period(df[args.period_col], args.freq),
            "flow": flows,
            "hs2": df[args.industry_col].str.replace(r"\D", "", regex=True).str.zfill(2).str[:2],
            "value_usd": pd.to_numeric(df[args.value_col].str.replace(",", ""), errors="coerce") * args.scale,
        }
    ).dropna(subset=["flow", "period"]).astype({"period": int})
    write_trade(finish_trade(raw, args.reporter.upper(), args.partner.upper()), Path(args.out_dir), args.freq,
                {"source": f"imported from {Path(args.file).name}", "freq": args.freq})


# --------------------------------------------------------------------------- market (optional)
def cmd_market(args):
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("Install yfinance first:  pip install yfinance")
    tickers = {**MACRO_TICKERS, **{e: e for e in sorted(reference_table()["equity_proxy"].dropna().unique())}}
    raw = yf.download(list(tickers), start=f"{args.start_year}-01-01", interval="1mo", auto_adjust=True, progress=False)
    close = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(how="all")
    close.index = pd.PeriodIndex(close.index, freq="M").strftime("%Y-%m")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    close.rename(columns=tickers).rename_axis("period").reset_index().to_csv(out_dir / "market_monthly.csv", index=False)
    print(f"Wrote {close.shape[1]} series -> {out_dir / 'market_monthly.csv'} (not used by scan.py yet)")


# --------------------------------------------------------------------------- CLI
def main():
    this_year = pd.Timestamp.today().year
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(default_out_dir()))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("comtrade", help="UN Comtrade bilateral trade, all HS-2 chapters")
    p.add_argument("--key", help="subscription key (or env COMTRADE_API_KEY)")
    p.add_argument("--freq", choices=["A", "M"], default="A")
    p.add_argument("--start-year", type=int, default=2005)
    p.add_argument("--end-year", type=int, default=this_year - 1)
    p.add_argument("--reporter", default="CAN", choices=sorted(CODES))
    p.add_argument("--partner", default="IND", choices=sorted(CODES))
    p.add_argument("--check", action="store_true", help="make one request and print the raw response")
    p.set_defaults(func=cmd_comtrade)

    p = sub.add_parser("worldbank", help="macro indicators for Canada and India from the World Bank")
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=this_year)
    p.set_defaults(func=cmd_worldbank)

    p = sub.add_parser("import-csv", help="standardise a CSV you downloaded")
    p.add_argument("file")
    p.add_argument("--period-col", required=True)
    p.add_argument("--industry-col", required=True, help="HS code column (any length; first 2 digits used)")
    p.add_argument("--flow-col", required=True)
    p.add_argument("--value-col", required=True)
    p.add_argument("--freq", choices=["A", "M"], default="A")
    p.add_argument("--flow-map", default="export=X,exports=X,import=M,imports=M",
                   help="comma list of file_value=X|M (X = Canada exports to India)")
    p.add_argument("--scale", type=float, default=1.0, help="multiply values, e.g. 1000000 if the file is in $ millions")
    p.add_argument("--reporter", default="CAN")
    p.add_argument("--partner", default="IND")
    p.set_defaults(func=cmd_import_csv)

    p = sub.add_parser("market", help="optional: FX, commodity, index and sector-ETF prices (yfinance)")
    p.add_argument("--start-year", type=int, default=2014)
    p.set_defaults(func=cmd_market)

    args = parser.parse_args()
    try:
        args.func(args)
    except ValueError as err:
        sys.exit(f"Error: {err}")


if __name__ == "__main__":
    main()
