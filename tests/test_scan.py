"""Tests for the all-industry scanner: industry reference, panel building, scan, data importers.

Run with:  pytest tests/test_scan.py
"""
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

import fetch_data
from build_panel import TARGET, build_panel, load_trade
from industries import CRITICAL_MINERAL_CHAPTERS, HS2_NAMES, reference_table, sector_of
from make_sample_data import make_trade_and_macro
from scan import run_scan
from train import chronological_split


@pytest.fixture(scope="module")
def sample():
    trade, macro = make_trade_and_macro()
    return trade, macro


# ----------------------------------------------------------------------------- industries
def test_every_chapter_has_a_sector_and_proxy():
    ref = reference_table()
    assert len(ref) == 96 and ref["hs2"].is_unique and "77" not in set(ref["hs2"])
    assert ref["sector"].notna().all() and ref["equity_proxy"].notna().all()
    assert sector_of("27") == sector_of(27) == "Mineral products"


def test_critical_mineral_flag_matches_notes():
    assert set(CRITICAL_MINERAL_CHAPTERS) <= set(HS2_NAMES)
    ref = reference_table().set_index("hs2")
    assert set(ref.index[ref["critical_mineral"]]) == set(CRITICAL_MINERAL_CHAPTERS)
    assert all(note.strip() for note in CRITICAL_MINERAL_CHAPTERS.values())


# ----------------------------------------------------------------------------- panel
def test_panel_covers_every_industry_and_both_flows(sample):
    trade, macro = sample
    panel = build_panel(trade, macro, min_value=0.0)
    assert panel.ppy == 1
    assert set(panel.train["hs2"]) == set(HS2_NAMES)  # every industry, not a hand-picked few
    assert set(panel.train["flow"]) == {"X", "M"}
    assert panel.train[TARGET].notna().all() and panel.latest[TARGET].isna().all()
    assert panel.latest["period"].nunique() == 1 and panel.latest["period"].iloc[0] == trade["period"].max()


def test_target_is_next_year_growth(sample):
    trade, _ = sample
    panel = build_panel(trade, None, min_value=0.0)
    df = panel.train.set_index(["series_id", "period"])
    sid = "X_27"
    t0 = int(df.loc[sid].index.min()) + 3
    now, nxt = df.loc[(sid, t0), "trailing_value_usd"], df.loc[(sid, t0 + 1), "trailing_value_usd"]
    expected = np.clip(np.log1p(nxt / 1e6) - np.log1p(now / 1e6), -3, 3)
    assert df.loc[(sid, t0), TARGET] == pytest.approx(expected)


@pytest.mark.parametrize("cut", [2008, 2012, 2016, 2021])
def test_features_never_use_the_future(sample, cut):
    """What the scanner sees at year `cut` must not depend on any later data.

    Build the panel from data up to `cut` only, and require that its latest rows (the ones the
    scanner would score) are identical - same series, same feature values - to the rows for that
    year in the panel built from the full history. A feature that peeks forward (even one year)
    is missing or different at the cut year, so this fails. (Comparing only rows present in both
    panels would miss that: the leaky rows simply drop out of the truncated panel.)
    """
    trade, macro = sample
    full = build_panel(trade, macro, min_value=0.0)
    early = build_panel(trade[trade["period"] <= cut], macro[macro["year"] <= cut], min_value=0.0)
    key = ["period", "series_id"]
    assert early.latest["period"].iloc[0] == cut
    both = pd.concat([full.train, full.latest])
    expected = both[both["period"] == cut].sort_values(key).set_index(key)[full.features]
    got = early.latest.sort_values(key).set_index(key)[full.features]
    assert len(got) > 100
    assert got.index.equals(expected.index)
    pd.testing.assert_frame_equal(got, expected, check_dtype=False, check_exact=False, rtol=1e-9)


def test_tiny_trade_base_is_dropped(sample):
    trade, _ = sample
    loose = build_panel(trade, None, min_value=0.0).train
    strict = build_panel(trade, None, min_value=1e8).train
    assert len(strict) < len(loose) and (strict["trailing_value_usd"] >= 1e8).all()


def test_mixed_country_pairs_are_rejected(sample, tmp_path):
    trade, _ = sample
    bad = pd.concat([trade, trade.assign(partner="USA")])
    path = tmp_path / "mixed.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="several reporter/partner pairs"):
        load_trade(path)


def test_too_little_history_is_a_clear_error(sample):
    trade, _ = sample
    with pytest.raises(ValueError, match="at least 4 years"):
        build_panel(trade[trade["period"] <= 2002], None)


def test_monthly_panel(tmp_path):
    rng = np.random.default_rng(5)
    months = pd.period_range("2016-01", "2024-12", freq="M")
    rows = []
    for hs2 in ["26", "27", "74", "85", "87"]:
        level = np.cumsum(rng.normal(0.004, 0.05, len(months))) + np.log(5e6)
        for m, v in zip(months, np.exp(level)):
            rows.append({"period": m.year * 100 + m.month, "reporter": "CAN", "partner": "IND",
                         "flow": "X", "hs2": hs2, "value_usd": v})
    panel = build_panel(pd.DataFrame(rows), None, min_value=0.0)
    assert panel.ppy == 12 and "mom_recent" in panel.features and "month" in panel.features
    assert panel.train[TARGET].notna().all()
    assert panel.latest["period"].iloc[0] == 202412


# ----------------------------------------------------------------------------- split
def test_chronological_split_respects_gap():
    t = np.repeat(np.arange(2000, 2020), 5)
    tr, te, info = chronological_split(t, test_size=0.25, gap=1)
    assert t[tr].max() + 1 < t[te].min()  # at least one whole period discarded between them
    assert set(t[tr]).isdisjoint(set(t[te])) and info["type"] == "chronological"
    with pytest.raises(ValueError):
        chronological_split(np.arange(5), test_size=0.5, gap=2)


# ----------------------------------------------------------------------------- scan
@pytest.fixture(scope="module")
def scan_result(sample, tmp_path_factory):
    trade, macro = sample
    keep = [f"{c:02d}" for c in range(1, 60)] + sorted(CRITICAL_MINERAL_CHAPTERS)
    trade = trade[trade["hs2"].isin(keep)]
    model_dir = tmp_path_factory.mktemp("scan") / "model"
    return run_scan(trade, macro, model_dir=model_dir, quick=True, min_value=1e6, verbose=False)


def test_scan_ranks_every_industry_exactly_once(scan_result):
    scan, panel = scan_result["scan"], scan_result["panel"]
    assert len(scan) == len(panel.latest) > 50
    assert scan["rank"].tolist() == list(range(1, len(scan) + 1))
    assert scan["series_id"].is_unique
    assert scan["signal"].is_monotonic_decreasing
    assert scan[["predicted_growth", "predicted_value_usd", "backtest_mae"]].notna().all().all()
    assert set(scan["flow"]) == {"exports", "imports"}
    assert scan["sector"].nunique() > 5  # many sectors, not one industry


def test_scan_critical_minerals_and_sectors(scan_result):
    scan, crit, sectors = scan_result["scan"], scan_result["critical"], scan_result["sectors"]
    assert set(crit["hs2"]) <= set(CRITICAL_MINERAL_CHAPTERS)
    flagged = scan[scan["critical_mineral"] == 1]
    assert len(flagged) == len(crit)
    assert (scan.loc[scan["critical_mineral"] == 1, "critical_mineral_note"] != "").all()
    assert sectors["industries"].sum() == len(scan)


def test_scan_backtest_is_honest_and_saved(scan_result):
    bt, model_dir = scan_result["backtest"], scan_result["model_dir"]
    names = {row["forecast"] for row in bt["table"]}
    assert "model" in names and any(n.startswith("naive") for n in names)
    assert isinstance(bt["model_beats_best_naive_rmse"], bool)
    reports = model_dir / "reports"
    for name in ["industry_scan.csv", "sector_summary.csv", "critical_minerals.csv", "backtest.json",
                 "test_predictions.csv"]:
        assert (reports / name).exists(), name
    assert json.loads((reports / "backtest.json").read_text())["n_test_periods"] >= 3
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert meta["split"]["type"] == "chronological"
    test_years = meta["split"]["test_periods"]
    train_years = meta["split"]["train_periods"]
    assert train_years[1] < test_years[0]  # train strictly before test


def test_rescoring_without_retraining(scan_result, sample):
    trade, macro = sample
    keep = [f"{c:02d}" for c in range(1, 60)] + sorted(CRITICAL_MINERAL_CHAPTERS)
    again = run_scan(trade[trade["hs2"].isin(keep)], macro, model_dir=scan_result["model_dir"],
                     retrain=False, verbose=False)
    pd.testing.assert_series_equal(again["scan"]["series_id"], scan_result["scan"]["series_id"])


# ----------------------------------------------------------------------------- data importers
def test_comtrade_parsing_and_cleaning():
    records = [
        {"period": 2023, "cmdCode": "27", "flowCode": "X", "primaryValue": 1000.0},
        {"period": "2023", "cmdCode": "2710", "flowCode": "X", "primaryValue": 500.0},  # longer code -> ch. 27
        {"period": 2023, "cmdCode": "TOTAL", "flowCode": "X", "primaryValue": 9e9},       # aggregate: dropped
        {"period": 2023, "cmdCode": "99", "flowCode": "M", "primaryValue": 7.0},          # not a chapter: dropped
        {"period": 2023, "cmdCode": "85", "flowCode": "M", "primaryValue": None, "cifvalue": 50.0},
        {"period": 2023, "cmdCode": "85", "flowCode": "RM", "primaryValue": 70.0},        # re-import: dropped
    ]
    raw = fetch_data.parse_comtrade_records(records, {"X": "X", "M": "M"})
    raw["period"] = fetch_data.parse_period(raw["period"], "A").astype(int)
    out = fetch_data.finish_trade(raw, "CAN", "IND")
    assert list(out.columns) == fetch_data.TRADE_COLUMNS
    got = {(r.flow, r.hs2): r.value_usd for r in out.itertuples()}
    assert got == {("X", "27"): 1500.0, ("M", "85"): 50.0}


def test_monthly_period_parsing():
    s = pd.Series(["202403", "2024-03", "Mar-2024", "2024-03-31"])
    assert fetch_data.parse_period(s, "M").astype(int).tolist() == [202403] * 4
    assert fetch_data.parse_period(pd.Series(["2023", 2023, "2023-12"]), "A").tolist() == [2023, 2023, 2023]


def test_import_csv_roundtrip_feeds_the_scanner(sample, tmp_path):
    trade, _ = sample
    # A "downloaded" file in a different layout: millions, comma thousands, long HS codes, text flows.
    df = pd.DataFrame(
        {
            "Year": trade["period"],
            "HS Code": trade["hs2"] + "0000",
            "Direction": trade["flow"].map({"X": "Exports", "M": "Imports"}),
            "Value ($M)": (trade["value_usd"] / 1e6).map("{:,.3f}".format),
        }
    )
    src = tmp_path / "download.csv"
    df.to_csv(src, index=False)
    args = Namespace(file=str(src), period_col="Year", industry_col="HS Code", flow_col="Direction",
                     value_col="Value ($M)", freq="A", flow_map="export=X,exports=X,import=M,imports=M",
                     scale=1e6, reporter="CAN", partner="IND", out_dir=str(tmp_path / "raw"))
    fetch_data.cmd_import_csv(args)
    out = load_trade(tmp_path / "raw" / "trade_CAN_IND_A.csv")
    assert set(out["hs2"]) == set(HS2_NAMES) and set(out["flow"]) == {"X", "M"}
    assert out["value_usd"].sum() == pytest.approx(trade["value_usd"].sum(), rel=1e-3)
    assert len(build_panel(out, None, min_value=0.0).train) > 1000


def test_import_csv_reports_bad_columns_and_flows(sample, tmp_path):
    trade, _ = sample
    src = tmp_path / "x.csv"
    trade.to_csv(src, index=False)
    base = dict(file=str(src), period_col="period", industry_col="hs2", flow_col="flow", value_col="value_usd",
                freq="A", flow_map="export=X,import=M", scale=1.0, reporter="CAN", partner="IND",
                out_dir=str(tmp_path / "raw"))
    with pytest.raises(SystemExit, match="not found"):
        fetch_data.cmd_import_csv(Namespace(**{**base, "period_col": "nope"}))
    with pytest.raises(SystemExit, match="matched no rows"):  # file uses X/M, flow-map expects words
        fetch_data.cmd_import_csv(Namespace(**base))


# ----------------------------------------------------------------------------- web app tab
def test_app_shows_the_scan(scan_result, monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    model_dir = scan_result["model_dir"]  # a trained model + scan reports in one folder
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("SCAN_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    root = Path(__file__).resolve().parent.parent
    at = AppTest.from_file(str(root / "app.py"), default_timeout=90).run()
    assert not at.exception
    assert any("Every industry, ranked" in s.value for s in at.subheader)
    # filter to critical-mineral chapters and make sure the page still renders
    at.checkbox[-1].check().run()
    assert not at.exception


def test_app_without_a_scan_shows_instructions(scan_result, monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MODEL_DIR", str(scan_result["model_dir"]))
    monkeypatch.setenv("SCAN_MODEL_DIR", str(tmp_path / "nothing"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    root = Path(__file__).resolve().parent.parent
    at = AppTest.from_file(str(root / "app.py"), default_timeout=90).run()
    assert not at.exception
    assert any("No scan yet" in i.value for i in at.info)
