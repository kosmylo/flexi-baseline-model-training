"""Command line: python -m plegma_baseline [--config CONFIG] <command> [options]."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m plegma_baseline",
                                 description="Plegma day-ahead household load forecasting baseline (FLEXI uUBI)")
    ap.add_argument("--config", help="YAML config (default: config.yaml in the project folder)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download and extract the clean archive")
    d.add_argument("--archive", help="use an archive that is already on disk")

    p = sub.add_parser("prepare", help="build hourly tables from the clean release")
    p.add_argument("--root", help="folder holding the House_<i> folders (found automatically after download)")

    v = sub.add_parser("devices", help="per-house device table with dataset wattage and measured power")
    v.add_argument("--root", help="folder holding the House_<i> folders (found automatically after download)")

    t = sub.add_parser("train", help="rolling-origin evaluation, final fit, saved run")
    t.add_argument("--run-name")
    t.add_argument("--compare-local", action="store_true", help="also train one model per house")
    t.add_argument("--cold-start", action="store_true", help="also score each house with a model trained on the others")
    t.add_argument("--extra-features", help="parquet with house, utc and extra feature columns")
    t.add_argument("--objective", help="override model.objective, for example l1, l2 or tweedie")

    e = sub.add_parser("export", help="copy a run into a portable model folder with a model card")
    e.add_argument("--run", required=True, help="run folder, e.g. runs/final_tweedie")
    e.add_argument("--name", required=True, help="model name, e.g. plegma-tweedie-v1")
    e.add_argument("--out", default="models", help="parent folder for the model (default: models)")

    f = sub.add_parser("predict", help="day-ahead forecast for one local day")
    f.add_argument("--run", required=True, help="run folder or exported model folder")
    f.add_argument("--date", required=True, help="local date, YYYY-MM-DD")
    f.add_argument("--house", type=int, action="append", help="repeat for several houses (default: all)")
    f.add_argument("--extra-features")
    f.add_argument("--out", help="write the forecast to this CSV")

    c = sub.add_parser("compare", help="paired comparison of two runs on the same folds")
    c.add_argument("--a", required=True, help="reference run, e.g. the baseline")
    c.add_argument("--b", required=True, help="candidate run, e.g. with assistant context")
    c.add_argument("--model-a", default="lgbm_global")
    c.add_argument("--model-b", default="lgbm_global")
    c.add_argument("--n-boot", type=int, default=2000)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    pd.set_option("display.width", 180)

    if args.cmd == "download":
        from .download import run_download
        print(run_download(cfg, args.archive))
    elif args.cmd == "prepare":
        from .prepare import prepare
        rep = prepare(cfg, args.root)
        print(json.dumps({k: rep[k] for k in ("clock_mode", "clock_check", "valid_share_all")}, indent=2))
    elif args.cmd == "devices":
        from .devices import build_device_table
        table = build_device_table(cfg, args.root)
        cols = ["house", "device", "wattage_w", "on_power_median_w", "on_power_p95_w",
                "runs_per_week", "run_energy_median_kwh", "run_duration_median_min"]
        print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    elif args.cmd == "train":
        from .train import run_training
        if args.objective:
            cfg["model"]["objective"] = args.objective
        run_dir = run_training(cfg, args.run_name, args.compare_local, args.cold_start, args.extra_features)
        print(pd.read_csv(run_dir / "metrics_overall.csv", index_col=0).round(3).to_string())
        print(f"\nrun saved to {run_dir}")
    elif args.cmd == "export":
        from .export import export_model
        print(f"model exported to {export_model(args.run, Path(args.out) / args.name)}")
    elif args.cmd == "predict":
        from .predict import forecast_day
        out = forecast_day(cfg, args.run, args.date, args.house, args.extra_features)
        if args.out:
            out.to_csv(args.out, index=False)
        print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    elif args.cmd == "compare":
        from .compare import compare_runs
        res = compare_runs(args.a, args.b, args.model_a, args.model_b, args.n_boot)
        print(json.dumps(res["summary"], indent=2))
        print(res["per_house"].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
