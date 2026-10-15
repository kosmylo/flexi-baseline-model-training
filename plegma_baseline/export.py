"""Turn a training run into a portable model folder for the repository (models/<name>)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

FILES = ("model.txt", "model_meta.json", "folds.json", "metrics_overall.csv", "metrics_by_house.csv",
         "metrics_by_month.csv", "feature_importance.csv")
HOUSES = Path(__file__).parent / "data" / "houses.csv"


def export_model(run_dir: str | Path, out_dir: str | Path) -> Path:
    """Copy the model, its metadata and scores, drop machine-specific paths, and write a model card."""
    run_dir, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(run_dir / name, out / name)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg.pop("paths", None)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copy2(HOUSES, out / "households.csv")
    meta = json.loads((out / "model_meta.json").read_text(encoding="utf-8"))
    (out / "MODEL_CARD.md").write_text(model_card(out.name, cfg, meta, out), encoding="utf-8")
    return out


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def model_card(name: str, cfg: dict, meta: dict, folder: Path) -> str:
    overall = pd.read_csv(folder / "metrics_overall.csv", index_col=0)
    folds = [f for f in json.loads((folder / "folds.json").read_text(encoding="utf-8")) if not f.get("skipped")]
    cols = ["wape", "daily_wape", "bias", "rmse_kwh", "skill_vs_seasonal_d7"]
    rows = ["| Forecast | Hourly WAPE | Daily WAPE | Bias | RMSE (kWh) | Skill vs last week |", "|---|---|---|---|---|---|"]
    labels = {"lgbm_global": "This model", "similar_day": "Similar-day average", "naive_d1": "Same hour yesterday",
              "seasonal_d7": "Same hour last week"}
    for key, label in labels.items():
        if key in overall.index:
            rows.append("| " + " | ".join([label] + [_fmt(float(overall.loc[key, c])) for c in cols]) + " |")
    start, end = meta.get("train_period_utc", ["", ""])
    fc = cfg["features"]
    inputs = "calendar (Greek holidays, Orthodox Easter week, tariff hours), same-hour load history, "
    inputs += "household descriptors" if fc.get("use_static", True) else ""
    inputs += ", outdoor weather" if fc.get("weather", "actual") != "none" else ""
    inputs += ", house ID" if fc.get("use_house_id", False) else ""
    return f"""# Model card: {name}

Day-ahead forecast of hourly whole-home electricity use. Trained on the Plegma dataset: 13 homes in Attica, Greece.

## Model

- LightGBM, objective `{meta.get('objective', cfg['model']['objective'])}`, {meta['final_rounds']} trees, {len(meta['features'])} inputs.
- Inputs: {inputs}.
- Target: hourly kWh divided by each home's mean load over the previous 28 days.
- Forecast made at local midnight (Europe/Athens) for the 24 hours of the next local day (23 or 25 on clock-change days).
- Package version {meta.get('package_version', '')}, LightGBM {meta['versions']['lightgbm']}.

## Training data

- Plegma clean release, homes {', '.join(str(h) for h in meta['houses'])}.
- {meta['train_hours']:,} training hours from {start[:10]} to {end[:10]} (UTC).
- Data: Athanasoulias et al., *Scientific Data* 11, 376 (2024), CC BY 4.0.

## Evaluation

Rolling origin with one test month per fold, {folds[0]['test_month']} to {folds[-1]['test_month']}. Each fold trains on all earlier data. All forecasts are scored on the same hours.

{chr(10).join(rows)}

## Intended use

- Reference baseline for FLEXI KPI1.1: the context-aware model reuses these settings and test months.
- Household-level demand forecasts for digital twins and simulated households.

## Scope notes

- Outdoor weather inputs take recorded values as a stand-in for a perfect day-ahead forecast. Real weather forecasts add some error.
- The model forecasts the mean hour. Short appliance bursts spread over neighbouring hours, which keeps energy totals close to reality.
- The scores describe 13 homes in Attica. Other climates and building stocks call for retraining with the same pipeline.
"""
