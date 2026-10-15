"""Load a saved model and forecast the next local day for one or more households.

Example:
    from plegma_baseline import Forecaster
    model = Forecaster.load("models/plegma-tweedie-v1")
    forecast = model.forecast(load, day="2023-09-20", weather=weather)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from . import features as F
from .train import predict_kwh

log = logging.getLogger(__name__)


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.as_unit("ns")


class Forecaster:
    """Day-ahead hourly household load forecaster: a saved LightGBM model plus its feature recipe.

    Inputs are plain DataFrames, so the same model serves Plegma homes, simulated households
    and any other home with an hourly meter series.
    """

    def __init__(self, booster: lgb.Booster, meta: dict, cfg: dict, households: pd.DataFrame | None = None):
        self.booster = booster
        self.meta = meta
        self.cfg = cfg
        self.households = households
        self.tz = cfg["prepare"]["timezone"]

    @classmethod
    def load(cls, path: str | Path) -> "Forecaster":
        """Load an exported model folder (models/<name>) or a training run folder (runs/<name>)."""
        path = Path(path)
        booster = lgb.Booster(model_file=str(path / "model.txt"))
        meta = json.loads((path / "model_meta.json").read_text(encoding="utf-8"))
        cfg = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
        hh = path / "households.csv"
        return cls(booster, meta, cfg, pd.read_csv(hh) if hh.exists() else F.load_static())

    @property
    def feature_names(self) -> list[str]:
        return list(self.meta["features"])

    @property
    def min_history_days(self) -> int:
        return int(self.cfg["features"]["scale_min_days"])

    def target_hours(self, day) -> pd.DatetimeIndex:
        """UTC hours of one local day: 24, or 23 and 25 on clock-change days."""
        day = pd.Timestamp(day).normalize()
        hours = pd.date_range(day, day + pd.Timedelta(days=1), freq="h", tz=self.tz, inclusive="left")
        return hours.tz_convert("UTC").as_unit("ns")

    def forecast(self, load: pd.DataFrame, day, weather: pd.DataFrame | None = None,
                 households: pd.DataFrame | None = None, extra: pd.DataFrame | None = None) -> pd.DataFrame:
        """Forecast the hourly energy (kWh) of every household in `load` for one local day.

        load        house, utc, load_kwh: hourly energy before `day`. At least `min_history_days` full
                    days are needed, 28 work best. Rows on or after `day` are ignored.
        weather     utc, ext_temp, ext_hum, optionally per house. Needed unless the model was trained
                    with weather: none. Target-day hours without values take the previous day's values.
        households  one row per house with the descriptors in plegma_baseline/data/houses.csv. Any
                    missing descriptor is allowed. Defaults to the households saved with the model.
        extra       house, utc and the extra feature columns, for models trained with --extra-features.

        Returns house, local_time, utc, forecast_kwh and two reference forecasts.
        """
        target = self.target_hours(day)
        load = load.loc[:, ["house", "utc", "load_kwh"]].copy()
        load["utc"] = _utc(load["utc"])
        houses = list(pd.unique(load["house"]))
        hist = load[load["utc"] < target[0]]
        grid = pd.MultiIndex.from_product([houses, target], names=["house", "utc"]).to_frame(index=False)
        grid["load_kwh"] = np.nan
        frame_in = pd.concat([hist, grid], ignore_index=True)

        env = None
        if self.cfg["features"].get("weather", "actual") != "none":
            if weather is None:
                raise ValueError("this model uses outdoor weather: pass weather with utc, ext_temp and ext_hum")
            env = self._weather(weather, houses, target)

        frame, feats, _ = F.build_frame(frame_in, env, households if households is not None else self.households,
                                        self.cfg, extra, heating=self.meta.get("heating_levels"))
        if feats != self.feature_names:
            raise ValueError("the inputs do not reproduce the model's features (check extra feature columns)")
        rows = frame[frame["utc"].isin(target)].copy()
        short = rows.loc[rows["scale"].isna(), "house"].unique()
        if len(short):
            log.warning("no forecast for %s: fewer than %d full days of history", list(short), self.min_history_days)
        rows["forecast_kwh"] = predict_kwh(self.booster, rows, feats)
        rows["local_time"] = rows["utc"].dt.tz_convert(self.tz)
        cols = ["house", "local_time", "utc", "forecast_kwh", "similar_day", "seasonal_d7"]
        return rows[cols].sort_values(["house", "utc"]).reset_index(drop=True)

    def _weather(self, weather: pd.DataFrame, houses: list, target: pd.DatetimeIndex) -> pd.DataFrame:
        w = weather.copy()
        w["utc"] = _utc(w["utc"])
        if "house" not in w.columns:  # one weather series for all households
            w = w.merge(pd.DataFrame({"house": houses}), how="cross")
        cols = ["house", "utc", "ext_temp", "ext_hum"]
        w = w.loc[w["house"].isin(houses), cols]
        grid = pd.MultiIndex.from_product([houses, target], names=["house", "utc"]).to_frame(index=False)
        cur = grid.merge(w, on=["house", "utc"], how="left")
        prev = grid.merge(w.assign(utc=w["utc"] + pd.Timedelta(hours=24)), on=["house", "utc"], how="left")
        missing = int(cur["ext_temp"].isna().sum())
        if missing:
            log.warning("%d target hours without weather take the previous day's values", missing)
        for c in ("ext_temp", "ext_hum"):
            cur[c] = cur[c].fillna(prev[c])
        rest = w.merge(grid.assign(_t=1), on=["house", "utc"], how="left")
        return pd.concat([rest[rest["_t"].isna()].drop(columns="_t"), cur], ignore_index=True)
