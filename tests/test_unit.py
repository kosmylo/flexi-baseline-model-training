"""Fast tests: calendar, run detection, metrics, and a small end-to-end round trip on synthetic data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plegma_baseline import Forecaster  # noqa: E402
from plegma_baseline import calendar_gr as cal  # noqa: E402
from plegma_baseline.config import load_config  # noqa: E402
from plegma_baseline.devices import _runs  # noqa: E402
from plegma_baseline.export import export_model  # noqa: E402
from plegma_baseline.metrics import evaluate, wape  # noqa: E402
from plegma_baseline.prepare import prepare  # noqa: E402
from plegma_baseline.train import run_training  # noqa: E402
from tests.make_synthetic import make  # noqa: E402


def test_orthodox_easter_and_holidays():
    assert cal.orthodox_easter(2023) == pd.Timestamp("2023-04-16")
    hol = cal.greek_holidays([2023])
    assert pd.Timestamp("2023-08-15") in hol and pd.Timestamp("2023-03-25") in hol


def test_tariff_windows():
    regimes = load_config(ROOT / "config.yaml")["features"]["tariff_regimes"]
    local = pd.Series(pd.to_datetime(["2023-01-10 03:00", "2023-01-10 12:00", "2023-07-10 23:30", "2025-07-10 12:00"]))
    assert cal.tariff_flag(local, regimes).tolist() == [True, False, True, True]


def test_runs_join_short_gaps_and_drop_short_runs():
    on = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0], dtype=bool)
    starts, ends = _runs(on, min_on=2, min_off=1)
    assert starts.tolist() == [1] and ends.tolist() == [6]


def test_optional_models_keep_the_headline_sample():
    rng = np.random.default_rng(0)
    n = 480
    pred = pd.DataFrame({"house": rng.integers(1, 3, n), "test_month": "2023-01", "y": rng.gamma(2, 0.3, n),
                         "local_date": pd.Timestamp("2023-01-01") + pd.to_timedelta(np.arange(n) // 24, unit="D")})
    for m in ("lgbm_global", "naive_d1", "seasonal_d7", "similar_day", "lgbm_local"):
        pred[m] = pred["y"] + rng.normal(0, 0.2, n)
    pred.loc[pred["house"] == 2, "lgbm_local"] = np.nan
    res = evaluate(pred, ["lgbm_global", "naive_d1", "seasonal_d7", "similar_day", "lgbm_local"])
    assert res["overall"].loc["lgbm_global", "n_hours"] == n
    sub = pred.dropna(subset=["lgbm_local"])
    paired = res["overall"].loc["lgbm_global [lgbm_local hours]", "wape"]
    assert paired == pytest.approx(wape(sub["y"], sub["lgbm_global"]))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    root = make(work / "data", n_houses=3, start="2023-01-01", end="2023-05-01", step_s=1800)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {"raw_dir": "raw", "processed_dir": "processed", "runs_dir": "runs"}
    cfg["prepare"]["clock"] = "auto"
    cfg["model"].update(learning_rate=0.2, num_leaves=15, max_rounds=100, early_stopping_rounds=10)
    cfg["cv"].update(first_test_month="2023-04")
    (work / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cfg = load_config(work / "cfg.yaml")
    prepare(cfg, root=root)
    return cfg, run_training(cfg, "t"), work


def test_clock_check_and_training(trained):
    cfg, run, _ = trained
    report = json.loads((Path(cfg["paths"]["processed_dir"]) / "prepare_report.json").read_text(encoding="utf-8"))
    assert report["clock_mode"] == "fixed_utc3"
    overall = pd.read_csv(run / "metrics_overall.csv", index_col=0)
    assert overall.loc["lgbm_global", "wape"] < overall.loc["seasonal_d7", "wape"]


def test_export_and_forecast(trained):
    cfg, run, work = trained
    folder = export_model(run, work / "models" / "test-model")
    assert (folder / "MODEL_CARD.md").exists()
    assert "paths" not in yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    model = Forecaster.load(folder)
    proc = Path(cfg["paths"]["processed_dir"])
    load = pd.read_parquet(proc / "hourly_load.parquet")
    weather = pd.read_parquet(proc / "hourly_env.parquet")
    out = model.forecast(load, day="2023-04-20", weather=weather)
    assert len(out) == 3 * 24 and out["forecast_kwh"].notna().all()
    # a household outside Plegma: its own ID, one weather series, partial descriptors
    own = load[load["house"] == 1].assign(house="my-home")
    one = weather[weather["house"] == 1].drop(columns="house")
    home = pd.DataFrame([{"house": "my-home", "occupants": 2, "n_ac": 1}])
    out2 = model.forecast(own, day="2023-04-20", weather=one, households=home)
    assert out2["house"].eq("my-home").all() and out2["forecast_kwh"].notna().all()
    # a clock-change day has 23 local hours
    assert len(model.forecast(load[load["house"] == 1], day="2023-03-26", weather=weather)) == 23
