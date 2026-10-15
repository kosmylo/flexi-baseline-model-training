"""Command-line day-ahead forecast from a training run or an exported model."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .forecaster import Forecaster


def forecast_day(cfg: dict, model_dir: str | Path, day: str, houses: list | None = None,
                 extra_path: str | None = None) -> pd.DataFrame:
    """Forecast one local day for the prepared Plegma tables (or a subset of houses)."""
    model = Forecaster.load(model_dir)
    proc = Path(cfg["paths"]["processed_dir"])
    load = pd.read_parquet(proc / "hourly_load.parquet")
    if houses:
        load = load[load["house"].isin(houses)]
    env_file = proc / "hourly_env.parquet"
    weather = pd.read_parquet(env_file) if env_file.exists() else None
    extra = pd.read_parquet(extra_path) if extra_path else None
    return model.forecast(load, day, weather=weather, extra=extra)
