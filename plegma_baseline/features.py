"""Features for day-ahead hourly forecasting with the forecast origin at local midnight.

Every feature for a target hour on local day D comes from days before D. The one exception is
outdoor weather in 'actual' mode, which stands in for a perfect day-ahead weather forecast.
The target is hourly energy divided by a trailing household scale, so one global model
can share patterns across homes of different size.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import calendar_gr as cal

KEYS = ["house", "utc"]
MAX_LAG = 14
BASELINES = ["naive_d1", "seasonal_d7", "similar_day"]
STATIC_NUM = ["occupants", "rooms", "built_year", "detached", "three_phase", "n_ac", "has_boiler", "has_dishwasher"]
HOUSE_LEVEL = {"house", "heating", *STATIC_NUM}
CALENDAR = ["hour", "dow", "is_day_off", "is_holiday", "month", "doy_sin", "doy_cos",
            "vacation", "easter_week", "tariff_window"]
DAILY = ["same_hour_mean7", "same_hour_std7", "prev_day_mean", "prev_day_max", "prev_day_min", "prev_week_mean"]
WEATHER = {
    "actual": ["ext_temp", "ext_hum", "temp_day_mean", "temp_day_max", "temp_day_min", "temp_prev_day_mean"],
    "persistence": ["ext_temp_d1", "ext_hum_d1", "temp_prev_day_mean", "temp_prev_day_max", "temp_prev_day_min"],
    "none": [],
}


def load_static(path: str | Path | None = None) -> pd.DataFrame:
    """Per-house descriptors transcribed from Tables 2 and 3 of the Plegma data descriptor."""
    return prepare_static(pd.read_csv(Path(path) if path else Path(__file__).parent / "data" / "houses.csv"))


def prepare_static(st: pd.DataFrame) -> pd.DataFrame:
    """Household descriptors in model form. Missing descriptors become NaN, which the model handles."""
    st = st.copy()
    if "detached" not in st.columns:
        dwelling = st["dwelling"] if "dwelling" in st.columns else pd.Series(np.nan, index=st.index)
        st["detached"] = np.where(dwelling.isna(), np.nan, (dwelling.astype(str).str.lower() == "detached").astype(float))
    for c in STATIC_NUM:
        st[c] = pd.to_numeric(st[c], errors="coerce") if c in st.columns else np.nan
    if "heating" not in st.columns:
        st["heating"] = np.nan
    return st


def heating_levels(static: pd.DataFrame) -> list[str]:
    """Heating types known to the model, in the order of their category codes."""
    return sorted(static["heating"].dropna().astype(str).unique())


def _house_key(s: pd.Series, numeric: bool) -> pd.Series:
    """House IDs as integers (Plegma) or strings (any other naming), so every merge matches."""
    return s.astype(int) if numeric else s.astype(str)


def _as_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.as_unit("ns")


def _lookup(table: pd.Series, house: np.ndarray, dates: pd.Series, hours: np.ndarray) -> np.ndarray:
    return table.reindex(pd.MultiIndex.from_arrays([house, dates, hours])).to_numpy()


def _daily_features(df: pd.DataFrame, fc: dict) -> pd.DataFrame:
    daily = df.groupby(["house", "local_date"]).agg(
        day_mean=("load_kwh", "mean"), day_max=("load_kwh", "max"),
        day_min=("load_kwh", "min"), n=("load_kwh", "count"))
    daily.loc[daily["n"] < fc["min_hours_per_day"], ["day_mean", "day_max", "day_min"]] = np.nan
    parts = []
    for h, g in daily.reset_index().groupby("house"):
        g = g.set_index("local_date")
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D").as_unit("ns"))
        prev = g["day_mean"].shift(1)
        parts.append(pd.DataFrame({
            "house": h,
            "local_date": g.index,
            "prev_day_mean": prev.to_numpy(),
            "prev_day_max": g["day_max"].shift(1).to_numpy(),
            "prev_day_min": g["day_min"].shift(1).to_numpy(),
            "prev_week_mean": prev.rolling(7, min_periods=4).mean().to_numpy(),
            "scale": prev.rolling(fc["scale_window_days"], min_periods=fc["scale_min_days"]).mean().to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def _weather(df: pd.DataFrame, env: pd.DataFrame, mode: str) -> pd.DataFrame:
    if env is None or not {"ext_temp", "ext_hum"} <= set(env.columns):
        raise ValueError("weather features need hourly_env.parquet with ext_temp and ext_hum (or set features.weather: none)")
    w = env.loc[:, ["house", "utc", "ext_temp", "ext_hum"]].copy()
    w["house"] = _house_key(w["house"], pd.api.types.is_integer_dtype(df["house"]))
    w["utc"] = _as_utc(w["utc"])
    df = df.merge(w.drop_duplicates(KEYS), on=KEYS, how="left")
    med = df.groupby("utc")[["ext_temp", "ext_hum"]].transform("median")  # all homes are in Attica
    df["ext_temp"] = df["ext_temp"].fillna(med["ext_temp"])
    df["ext_hum"] = df["ext_hum"].fillna(med["ext_hum"])
    dw = df.groupby(["house", "local_date"])["ext_temp"].agg(
        temp_day_mean="mean", temp_day_max="max", temp_day_min="min").reset_index()
    pv = dw.rename(columns=lambda c: c.replace("temp_day_", "temp_prev_day_"))
    pv["local_date"] = pv["local_date"] + pd.Timedelta(days=1)
    df = df.merge(dw, on=["house", "local_date"], how="left").merge(pv, on=["house", "local_date"], how="left")
    if mode == "persistence":
        house, hour = df["house"].to_numpy(), df["hour"].to_numpy()
        for c in ("ext_temp", "ext_hum"):
            t = df.groupby(["house", "local_date", "hour"])[c].mean()
            df[f"{c}_d1"] = _lookup(t, house, df["local_date"] - pd.Timedelta(days=1), hour)
    return df


def build_frame(load: pd.DataFrame, env: pd.DataFrame | None, static: pd.DataFrame | None,
                cfg: dict, extra: pd.DataFrame | None = None, heating: list[str] | None = None):
    """Return (frame, feature_names, categorical_feature_names).

    `heating` fixes the category codes of the heating type. Pass the list saved with a model
    so that new households get the same codes as in training.
    """
    fc, tz = cfg["features"], cfg["prepare"]["timezone"]
    df = load.loc[:, ["house", "utc", "load_kwh"]].copy()
    numeric = pd.api.types.is_numeric_dtype(df["house"])
    if fc.get("use_house_id", False) and not numeric:
        raise ValueError("features.use_house_id needs integer house IDs")
    df["house"] = _house_key(df["house"], numeric)
    df["utc"] = _as_utc(df["utc"])
    df = df.sort_values(KEYS).drop_duplicates(KEYS).reset_index(drop=True)

    # calendar, on the Athens wall clock
    local = df["utc"].dt.tz_convert(tz).dt.tz_localize(None)
    df["local_date"] = local.dt.normalize()
    df["hour"] = local.dt.hour
    df["dow"] = local.dt.dayofweek
    df["month"] = local.dt.month
    doy = local.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    hol = cal.greek_holidays(range(int(local.dt.year.min()) - 1, int(local.dt.year.max()) + 2))
    df["is_holiday"] = df["local_date"].isin(hol).astype(int)
    df["is_day_off"] = cal.is_day_off(df["local_date"], hol).astype(int)
    df["vacation"] = cal.in_periods(df["local_date"], fc.get("vacation_periods")).astype(int)
    df["easter_week"] = cal.easter_week(df["local_date"]).astype(int)
    df["tariff_window"] = cal.tariff_flag(local, fc.get("tariff_regimes")).astype(int)

    # same local hour on each of the previous 14 days
    table = df.groupby(["house", "local_date", "hour"])["load_kwh"].mean()
    house, hour = df["house"].to_numpy(), df["hour"].to_numpy()
    for k in range(1, MAX_LAG + 1):
        df[f"lag_d{k}"] = _lookup(table, house, df["local_date"] - pd.Timedelta(days=k), hour)
    week = df[[f"lag_d{k}" for k in range(1, 8)]].to_numpy()
    enough = np.isfinite(week).sum(axis=1) >= 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df["same_hour_mean7"] = np.where(enough, np.nanmean(week, axis=1), np.nan)
        df["same_hour_std7"] = np.where(enough, np.nanstd(week, axis=1), np.nan)

    # previous-day statistics and the trailing household scale
    df = df.merge(_daily_features(df, fc), on=["house", "local_date"], how="left")
    df.loc[~(df["scale"] > 0), "scale"] = np.nan
    df["log_scale"] = np.log(df["scale"])

    mode = fc.get("weather", "actual")
    if mode != "none":
        df = _weather(df, env, mode)

    cats: list[str] = []
    if fc.get("use_static", True) and static is not None:
        st = prepare_static(static)
        codes = {v: i for i, v in enumerate(heating if heating is not None else heating_levels(st))}
        st["heating"] = st["heating"].map(lambda v: codes.get(str(v), -1) if pd.notna(v) else -1)
        st["house"] = _house_key(st["house"], numeric)
        df = df.merge(st[["house", *STATIC_NUM, "heating"]].drop_duplicates("house"), on="house", how="left")
        df["heating"] = df["heating"].fillna(-1).astype(int)

    # normalized inputs and target
    norm = [f"lag_d{k}" for k in fc["lag_days"]] + DAILY
    for c in norm:
        df[f"{c}_n"] = df[c] / df["scale"]
    df["y"] = df["load_kwh"]
    df["y_n"] = df["y"] / df["scale"]

    # reference forecasts in kWh
    df["naive_d1"] = df["lag_d1"]
    df["seasonal_d7"] = df["lag_d7"]
    target_off = df["is_day_off"].to_numpy().astype(bool)
    acc, cnt = np.zeros(len(df)), np.zeros(len(df))
    for k in range(1, MAX_LAG + 1):  # mean of the last (up to 5) days of the same type
        v = df[f"lag_d{k}"].to_numpy()
        same = cal.is_day_off(df["local_date"] - pd.Timedelta(days=k), hol) == target_off
        m = same & np.isfinite(v) & (cnt < 5)
        acc[m] += v[m]
        cnt[m] += 1
    df["similar_day"] = np.where(cnt >= 2, acc / np.maximum(cnt, 1), np.nan)

    extra_cols: list[str] = []
    if extra is not None:
        ex = extra.copy()
        ex["house"] = _house_key(ex["house"], numeric)
        ex["utc"] = _as_utc(ex["utc"])
        extra_cols = [c for c in ex.columns if c not in KEYS]
        clash = sorted(set(extra_cols) & set(df.columns))
        if clash:
            raise ValueError(f"extra feature names clash with built-in columns: {clash}")
        df = df.merge(ex.drop_duplicates(KEYS), on=KEYS, how="left")

    df["train_ok"] = np.isfinite(df["y_n"].to_numpy(dtype=float))
    df["eval_ok"] = df["train_ok"] & df[BASELINES].notna().all(axis=1)

    feats = CALENDAR + [f"{c}_n" for c in norm] + ["log_scale"] + WEATHER[mode]
    if "heating" in df.columns:
        feats += [*STATIC_NUM, "heating"]
        cats.append("heating")
    if fc.get("use_house_id", False):
        feats.append("house")
        cats.append("house")
    feats += extra_cols
    return df, feats, cats
