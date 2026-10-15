"""Checks on the prepared tables: which clock the timestamps use, and how long each house is covered.

Clock. At a DST change the sun keeps its schedule while people move with the wall clock.
On a fixed UTC+3 clock the outdoor temperature cycle stays put and daily routines move by one hour.
On a local wall clock it is the other way round. The check rebuilds the clock of the CSV files from the
prepared tables and measures both shifts around each change. AC and boiler load are left out of the
routine signal, because heating and the Greek night tariff (new schedule on 1 November) also move them.

Run from the project folder:  python -m plegma_baseline.diagnose
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

TZ = "Europe/Athens"
# Shift of daily routines on a fixed UTC+3 clock, in hours (later is positive)
TRANSITIONS = (("2022-10-30", 1), ("2023-03-26", -1), ("2023-10-29", 1))
HEAVY = r"(?i)^((ac|air_?cond\w*)(_?\d+)?|(boiler|water_?heater)(_?\d+)?)$"


def raw_clock(utc: pd.Series, mode: str) -> pd.Series:
    """Timestamps on the clock of the CSV files, rebuilt from the UTC times written by prepare."""
    utc = pd.to_datetime(utc, utc=True)
    if mode == "fixed_utc3":
        return (utc + pd.Timedelta(hours=3)).dt.tz_localize(None)
    if mode == "local":
        return utc.dt.tz_convert(TZ).dt.tz_localize(None)
    return utc.dt.tz_localize(None)


def _profile(df: pd.DataFrame, col: str) -> tuple[np.ndarray | None, int]:
    """Median over houses of each house's standardized mean profile by hour of day."""
    profiles = []
    for _, g in df.dropna(subset=[col]).groupby("house"):
        if g["raw"].dt.normalize().nunique() < 7:
            continue
        p = g.groupby(g["raw"].dt.hour)[col].mean().reindex(range(24)).interpolate(limit_direction="both")
        if p.isna().any() or p.std() == 0:
            continue
        profiles.append(((p - p.mean()) / p.std()).to_numpy())
    return (np.median(profiles, axis=0), len(profiles)) if profiles else (None, 0)


def _best_shift(before: np.ndarray, after: np.ndarray) -> tuple[int, dict]:
    """Hours by which the after-profile runs later than the before-profile."""
    dist = {s: float(np.abs(before - np.roll(after, -s)).mean()) for s in (-2, -1, 0, 1, 2)}
    return min(dist, key=dist.get), dist


def check_clock(load: pd.DataFrame, apps: pd.DataFrame | None, env: pd.DataFrame | None, mode: str):
    routine = load[["house", "utc", "load_kwh"]].copy()
    if apps is not None and len(apps):
        heavy = apps[apps["appliance"].astype(str).str.match(HEAVY)]
        heavy = heavy.groupby(["house", "utc"], as_index=False)["kwh"].sum().rename(columns={"kwh": "heavy"})
        routine = routine.merge(heavy, on=["house", "utc"], how="left")
        routine["load_kwh"] = (routine["load_kwh"] - routine["heavy"].fillna(0)).clip(lower=0)
    routine["raw"] = raw_clock(routine["utc"], mode)
    signals = {"routine load": (routine, "load_kwh")}
    if env is not None and "ext_temp" in env.columns:
        weather = env[["house", "utc", "ext_temp"]].copy()
        weather["raw"] = raw_clock(weather["utc"], mode)
        signals["outdoor temperature"] = (weather, "ext_temp")
    rows = []
    for day, routine_shift in TRANSITIONS:
        t0 = pd.Timestamp(day)
        for name, (df, col) in signals.items():
            before = df[(df["raw"] >= t0 - pd.Timedelta(days=14)) & (df["raw"] < t0)]
            after = df[(df["raw"] >= t0 + pd.Timedelta(days=1)) & (df["raw"] < t0 + pd.Timedelta(days=15))]
            pb, nb = _profile(before, col)
            pa, na = _profile(after, col)
            if pb is None or pa is None:
                continue
            shift, dist = _best_shift(pb, pa)
            expected = {"fixed_utc3": routine_shift if name == "routine load" else 0,
                        "local": 0 if name == "routine load" else -routine_shift}
            fits = [k for k, v in expected.items() if v == shift]
            rows.append({"transition": day, "signal": name, "houses": min(nb, na), "shift_h": shift,
                         "dist_-1h": dist[-1], "dist_0h": dist[0], "dist_+1h": dist[1],
                         "fits": fits[0] if fits else "neither"})
    table = pd.DataFrame(rows)
    verdict = None
    if len(table):
        votes = table["fits"].value_counts()
        for k in ("fixed_utc3", "local"):
            if votes.get(k, 0) == len(table):
                verdict = k
            elif votes.get(k, 0) >= 0.75 * len(table) and votes.get("fixed_utc3" if k == "local" else "local", 0) == 0:
                verdict = f"probably {k}"
    return table, verdict


def coverage(load: pd.DataFrame) -> pd.DataFrame:
    local = pd.to_datetime(load["utc"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    df = load.assign(day=local.dt.normalize(), month=local.dt.to_period("M"))
    valid_days = df[df["valid"]].groupby(["house", "day"]).size()
    out = df.groupby("house").agg(first_day=("day", "min"), last_day=("day", "max"), months=("month", "nunique"),
                                  valid_share=("valid", "mean"))
    out["full_days"] = (valid_days >= 18).groupby(level="house").sum()
    out["first_day"] = out["first_day"].dt.date
    out["last_day"] = out["last_day"].dt.date
    return out.round({"valid_share": 3})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Clock and coverage checks on the prepared tables")
    ap.add_argument("--config", help="YAML config (default: config.yaml in the project folder)")
    args = ap.parse_args(argv)
    proc = Path(load_config(args.config)["paths"]["processed_dir"])
    mode = json.loads((proc / "prepare_report.json").read_text(encoding="utf-8"))["clock_mode"]
    load = pd.read_parquet(proc / "hourly_load.parquet")
    apps_file, env_file = proc / "hourly_appliances.parquet", proc / "hourly_env.parquet"
    apps = pd.read_parquet(apps_file) if apps_file.exists() else None
    env = pd.read_parquet(env_file) if env_file.exists() else None
    pd.set_option("display.width", 160)
    print("Coverage per house (Athens dates, full days have at least 18 valid hours)")
    print(coverage(load).to_string())
    table, verdict = check_clock(load, apps, env, mode)
    print("\nShift after each DST change, in hours (later is positive), with the profile distance per shift")
    print(table.round(3).to_string(index=False) if len(table) else "no house has 7 days on both sides of a change")
    print("\nFixed UTC+3 clock: outdoor temperature 0 h, routine load +1 h in autumn and -1 h in spring.")
    print("Local clock: outdoor temperature -1 h in autumn and +1 h in spring, routine load 0 h.")
    print(f"Verdict: {verdict or 'inconclusive'} (prepare used '{mode}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
