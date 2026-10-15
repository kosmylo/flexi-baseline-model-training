"""Paired comparison of two runs on identical folds, for example baseline against a context-aware model (KPI1.1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import wape


def _folds(run: Path) -> list[tuple[str, str]]:
    folds = json.loads((run / "folds.json").read_text(encoding="utf-8"))
    return [(f["test_start"], f["test_end"]) for f in folds if not f.get("skipped")]


def compare_runs(run_a, run_b, model_a: str = "lgbm_global", model_b: str = "lgbm_global",
                 n_boot: int = 2000, seed: int = 0) -> dict:
    """Relative WAPE reduction of B against A, with a bootstrap CI that resamples house-days."""
    run_a, run_b = Path(run_a), Path(run_b)
    if _folds(run_a) != _folds(run_b):
        raise ValueError("The runs used different test folds. Use the same cv settings for both.")
    a = pd.read_parquet(run_a / "predictions.parquet")[["house", "utc", "local_date", "y", model_a]]
    b = pd.read_parquet(run_b / "predictions.parquet")[["house", "utc", model_b]]
    m = a.rename(columns={model_a: "pa"}).merge(b.rename(columns={model_b: "pb"}), on=["house", "utc"])
    m = m.dropna(subset=["y", "pa", "pb"])
    days = m.assign(ea=(m["pa"] - m["y"]).abs(), eb=(m["pb"] - m["y"]).abs()) \
            .groupby(["house", "local_date"])[["ea", "eb"]].sum()
    ea, eb = days["ea"].to_numpy(), days["eb"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(days), size=(n_boot, len(days)), dtype=np.int32)
    boot = 1.0 - eb[idx].sum(axis=1) / ea[idx].sum(axis=1)
    per_house = pd.DataFrame({h: {"wape_a": wape(g["y"], g["pa"]), "wape_b": wape(g["y"], g["pb"])}
                              for h, g in m.groupby("house")}).T
    per_house.index.name = "house"
    per_house["relative_reduction"] = 1 - per_house["wape_b"] / per_house["wape_a"]
    wa, wb = wape(m["y"], m["pa"]), wape(m["y"], m["pb"])
    summary = {
        "run_a": str(run_a), "model_a": model_a, "run_b": str(run_b), "model_b": model_b,
        "hours": int(len(m)), "house_days": int(len(days)),
        "wape_a": round(wa, 4), "wape_b": round(wb, 4),
        "relative_reduction": round(1 - wb / wa, 4),
        "ci95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "houses_improved": int((per_house["relative_reduction"] > 0).sum()),
        "houses": int(len(per_house)),
    }
    return {"summary": summary, "per_house": per_house}
