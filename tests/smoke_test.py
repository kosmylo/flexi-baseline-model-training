"""End-to-end smoke test on synthetic data.

Covers archive extraction, the clock check, hourly preparation, rolling-origin training with the
local and cold-start comparisons, saving, day-ahead prediction, and a paired comparison against a
run with extra (oracle) context features. Run from the project folder:

    python tests/smoke_test.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plegma_baseline import Forecaster  # noqa: E402
from plegma_baseline.__main__ import main as cli  # noqa: E402
from plegma_baseline.config import load_config  # noqa: E402
from plegma_baseline.download import run_download  # noqa: E402
from plegma_baseline.prepare import prepare  # noqa: E402
from tests.make_synthetic import make  # noqa: E402

WORK = ROOT / ".smoke"


def _config(name: str, overrides: dict) -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {"raw_dir": f"{name}/raw", "processed_dir": f"{name}/processed", "runs_dir": f"{name}/runs"}
    cfg["prepare"]["clock"] = "auto"  # the smoke test exercises the automatic clock check
    for section, values in overrides.items():
        cfg[section].update(values)
    path = WORK / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return load_config(path)


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def main() -> None:
    t0 = time.time()
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir()
    # small, fast model settings: this test checks the plumbing, not the accuracy
    fast = {"model": {"learning_rate": 0.2, "num_leaves": 15, "max_rounds": 300, "early_stopping_rounds": 30},
            "cv": {"first_test_month": "2023-08"}}

    # 1. synthetic clean release, packed like the real archive (7z)
    src = make(WORK / "synthetic", step_s=600, clock="fixed_utc3")
    archive = WORK / "PlegmaDataset_Clean.7z"
    with py7zr.SevenZipFile(archive, "w", filters=[{"id": py7zr.FILTER_LZMA2, "preset": 1}]) as z:
        z.writeall(src, "PlegmaDataset_Clean/Clean_dataset")
    cfg = _config("main", fast)
    root = run_download(cfg, str(archive))
    check(root.name == "Clean_dataset", f"extraction and root detection ({root})")

    # 2. hourly tables and clock check (synthetic data written on a fixed UTC+3 clock)
    rep = prepare(cfg)
    check(rep["clock_mode"] == "fixed_utc3", f"clock check detects fixed UTC+3 ({rep['clock_check']})")
    load = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "hourly_load.parquet")
    check(load["house"].nunique() == 13, "13 houses in hourly_load")

    # 3. train with local and cold-start comparisons, evaluate and save
    cli(["--config", str(WORK / "main.yaml"), "train", "--run-name", "baseline", "--compare-local", "--cold-start"])
    run = Path(cfg["paths"]["runs_dir"]) / "baseline"
    for f in ("model.txt", "model_meta.json", "predictions.parquet", "metrics_overall.csv",
              "metrics_by_house.csv", "metrics_by_month.csv", "folds.json", "feature_importance.csv", "config.yaml"):
        check((run / f).exists(), f"run artifact {f}")
    ov = pd.read_csv(run / "metrics_overall.csv", index_col=0)
    check(ov.loc["lgbm_global", "wape"] < ov.loc["seasonal_d7", "wape"], "global model beats seasonal naive")
    check({"lgbm_local", "lgbm_coldstart"} <= set(ov.index), "per-house and cold-start models scored")
    folds = json.loads((run / "folds.json").read_text(encoding="utf-8"))
    check(any(f.get("skipped") for f in folds), "partial last month skipped")

    # 4. day-ahead forecast inside the data and for the day after it ends (weather by persistence)
    cli(["--config", str(WORK / "main.yaml"), "predict", "--run", str(run), "--date", "2023-09-20",
         "--house", "1", "--out", str(WORK / "fc_inside.csv")])
    fc = pd.read_csv(WORK / "fc_inside.csv")
    check(len(fc) == 24 and fc["forecast_kwh"].notna().all(), "24 hourly forecasts for one house")
    cli(["--config", str(WORK / "main.yaml"), "predict", "--run", str(run), "--date", "2023-10-01",
         "--out", str(WORK / "fc_next.csv")])
    check(pd.read_csv(WORK / "fc_next.csv")["forecast_kwh"].notna().mean() > 0.9, "forecast for the day after the data")

    # 5. export a portable model folder and use it through the Python API
    cli(["--config", str(WORK / "main.yaml"), "export", "--run", str(run), "--name", "smoke-model",
         "--out", str(WORK / "models")])
    model = Forecaster.load(WORK / "models" / "smoke-model")
    proc = Path(cfg["paths"]["processed_dir"])
    out = model.forecast(pd.read_parquet(proc / "hourly_load.parquet"), day="2023-09-20",
                         weather=pd.read_parquet(proc / "hourly_env.parquet"))
    check(out["forecast_kwh"].notna().all(), "exported model forecasts through the Python API")

    # 6. context hook: oracle 'declared plans' from flexible appliances, 80% recall per house-day
    app = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "hourly_appliances.parquet")
    app = app[app["appliance"].str.match(r"(ac_\d+|boiler|washing_machine|washer_dryer|dishwasher)$")].copy()
    app["day"] = pd.to_datetime(app["utc"]).dt.tz_convert("Europe/Athens").dt.date
    keys = app[["house", "day", "appliance"]].drop_duplicates()
    keys["keep"] = np.random.default_rng(0).random(len(keys)) < 0.8
    app = app.merge(keys, on=["house", "day", "appliance"])
    app["declared"] = app["kwh"].fillna(0) * app["keep"]
    ctx = app.groupby(["house", "utc"], as_index=False)["declared"].sum().rename(columns={"declared": "ctx_declared_kwh"})
    ctx.to_parquet(WORK / "context.parquet", index=False)
    cli(["--config", str(WORK / "main.yaml"), "train", "--run-name", "context",
         "--extra-features", str(WORK / "context.parquet")])
    cli(["--config", str(WORK / "main.yaml"), "compare", "--a", str(run),
         "--b", str(Path(cfg["paths"]["runs_dir"]) / "context"), "--n-boot", "500"])

    # 7. 10-second files and a local wall clock
    make(WORK / "tensec", n_houses=2, start="2022-10-10", end="2022-11-14", step_s=10, clock="local")
    cfg10 = _config("tensec_run", {})
    rep10 = prepare(cfg10, root=WORK / "tensec")
    steps = {v["sample_step_s"] for v in rep10["houses"].values()}
    check(steps == {10.0}, f"10 s sampling detected ({steps})")
    check(rep10["clock_mode"] == "local", f"clock check detects a local wall clock ({rep10['clock_check']})")
    cli(["--config", str(WORK / "tensec_run.yaml"), "devices", "--root", str(WORK / "tensec")])
    dev = pd.read_csv(Path(cfg10["paths"]["processed_dir"]) / "devices.csv")
    check(dev["on_power_median_w"].notna().all(), "device table with measured on-power")
    print(json.dumps({"elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
