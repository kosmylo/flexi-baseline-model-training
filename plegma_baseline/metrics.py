"""Error metrics. WAPE (total absolute error over total consumption) is the headline metric."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.abs(y).sum()
    return float(np.abs(y - p).sum() / d) if d > 0 else float("nan")


def _scores(d: pd.DataFrame, m: str, min_hours_day: int) -> dict:
    y, p = d["y"].to_numpy(float), d[m].to_numpy(float)
    e = p - y
    days = d.groupby(["house", "local_date"]).agg(y=("y", "sum"), p=(m, "sum"), n=("y", "size"))
    days = days[days["n"] >= min_hours_day]
    per_house = [wape(g["y"], g[m]) for _, g in d.groupby("house")]
    return {
        "wape": wape(y, p),
        "wape_house_mean": float(np.nanmean(per_house)),
        "daily_wape": wape(days["y"], days["p"]),
        "mae_kwh": float(np.abs(e).mean()),
        "rmse_kwh": float(np.sqrt((e ** 2).mean())),
        "bias": float(e.sum() / y.sum()) if y.sum() > 0 else float("nan"),
        "n_hours": int(len(d)),
        "n_days": int(len(days)),
    }


CORE = ("lgbm_global", "naive_d1", "seasonal_d7", "similar_day")


def evaluate(pred: pd.DataFrame, models: list[str], reference: str = "seasonal_d7",
             min_hours_day: int = 20) -> dict:
    """Score the global model and the references on the hours they share.

    Optional models (per-house, cold-start) cover fewer hours. Each is scored on its own hours,
    next to the global model on exactly those hours, so the headline sample stays the same.
    """
    core = [m for m in models if m in CORE]
    common = pred.dropna(subset=["y", *core])
    frames = {m: (m, common) for m in core}
    for m in (m for m in models if m not in CORE):
        sub = common.dropna(subset=[m])
        if len(sub):
            frames[m] = (m, sub)
            frames[f"lgbm_global [{m} hours]"] = ("lgbm_global", sub)
    overall = pd.DataFrame({
        name: {**_scores(d, col, min_hours_day),
               f"skill_vs_{reference}": 1 - wape(d["y"], d[col]) / wape(d["y"], d[reference])}
        for name, (col, d) in frames.items()}).T
    overall.index.name = "model"
    by_house = pd.DataFrame({name: {h: wape(g["y"], g[col]) for h, g in d.groupby("house")}
                             for name, (col, d) in frames.items()})
    by_house.index.name = "house"
    by_month = pd.DataFrame({name: {k: wape(g["y"], g[col]) for k, g in d.groupby("test_month")}
                             for name, (col, d) in frames.items()})
    by_month.index.name = "test_month"
    return {"overall": overall, "by_house": by_house, "by_month": by_month, "n_common": int(len(common))}
