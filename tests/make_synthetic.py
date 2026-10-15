"""Synthetic replica of the Plegma clean release, for smoke tests.

The folder layout, file names, column names and timestamp format follow the data descriptor.
All values are synthetic. Routines follow the Athens wall clock and timestamps are written on
the clock chosen with `clock`, so the clock check in `prepare` can be exercised.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plegma_baseline import calendar_gr as cal  # noqa: E402

TZ = "Europe/Athens"
FMT = "%m/%d/%Y %I:%M:%S %p"


def _bump(h: np.ndarray, center, width: float) -> np.ndarray:
    d = (h - center + 12) % 24 - 12
    return np.exp(-0.5 * (d / width) ** 2)


def _events(days: pd.Series, rng, prob: float, lo: float, hi: float) -> np.ndarray:
    """Start hour of one event on a share of days, NaN on the other days."""
    uniq = pd.Index(days.unique())
    starts = np.where(rng.random(len(uniq)) < prob, rng.uniform(lo, hi, len(uniq)), np.nan)
    return days.map(pd.Series(starts, index=uniq)).to_numpy()


def _cycle(t_rel: np.ndarray, phases) -> np.ndarray:
    """Piecewise power profile: phases = [(duration_h, watts), ...] from the event start."""
    out = np.zeros(len(t_rel))
    t0 = 0.0
    for dur, watts in phases:
        out = np.where((t_rel >= t0) & (t_rel < t0 + dur), watts, out)
        t0 += dur
    return out


def make(out: str | Path, n_houses: int = 13, start: str = "2022-07-15", end: str = "2023-10-01",
         step_s: int = 300, clock: str = "fixed_utc3", seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    static = pd.read_csv(ROOT / "plegma_baseline" / "data" / "houses.csv").set_index("house")
    utc = pd.date_range(start, end, freq=f"{step_s}s", tz="UTC", inclusive="left")
    local = utc.tz_convert(TZ).tz_localize(None)
    raw = (utc.tz_localize(None) + pd.Timedelta(hours=3)) if clock == "fixed_utc3" else local
    stamp = np.asarray(raw.strftime(FMT))
    month = np.asarray(raw.strftime("%Y-%m"))
    h = np.asarray(local.hour + local.minute / 60.0)
    days = pd.Series(local.normalize())
    day_idx = pd.factorize(days)[0]
    doy = np.asarray(local.dayofyear)
    hol = cal.greek_holidays(range(local.year.min(), local.year.max() + 1))
    off = cal.is_day_off(days, hol).astype(float)

    anomaly = np.cumsum(rng.normal(0, 0.9, day_idx.max() + 1))
    window = min(21, len(anomaly))
    anomaly -= np.convolve(anomaly, np.ones(window) / window, mode="same")
    solar = np.asarray(utc.hour + utc.minute / 60.0) + 2.0  # the weather follows the sun, not the wall clock
    temp = (19 + 8.5 * np.sin(2 * np.pi * (doy - 110) / 365.25) + 4.5 * np.sin(2 * np.pi * (solar - 9) / 24)
            + anomaly[day_idx] + rng.normal(0, 0.4, len(h)))
    hum = np.clip(62 - 1.4 * (temp - 19) + rng.normal(0, 4, len(h)), 15, 98)
    env_mask = np.asarray((utc.minute % 15 == 0) & (utc.second == 0))

    base_dir = Path(out) / "Clean_dataset"
    for house in range(1, n_houses + 1):
        st = static.loc[house]
        r = np.random.default_rng(seed + house)
        hd = base_dir / f"House_{house}"
        (hd / "Electric_data").mkdir(parents=True, exist_ok=True)
        (hd / "Environmental_data").mkdir(parents=True, exist_ok=True)
        occ = float(st["occupants"])
        wake = 7.0 + 0.5 * r.standard_normal() + 1.5 * off
        routine = (380 * np.sqrt(occ) * _bump(h, wake + 0.7, 0.8)
                   + 650 * np.sqrt(occ) * _bump(h, 20.5 + 0.3 * r.standard_normal(), 1.8)
                   + 300 * off * _bump(h, 14, 1.8))
        cols: dict[str, np.ndarray] = {}
        ac_total = np.zeros(len(h))
        evening = ((h >= 15) | (h < 1)).astype(float)
        for i in range(int(st["n_ac"])):
            cool = 700 * np.clip((temp - 26.5) / 6, 0, 1.4) * evening * (0.6 + 0.4 * r.random())
            heat = 750 * np.clip((12.5 - temp) / 8, 0, 1) * (_bump(h, 21, 2.2) + 0.6 * _bump(h, 8, 1.0))
            cols[f"ac_{i + 1}"] = cool + heat
            ac_total += cool + heat
        if st["has_boiler"]:
            start_h = np.where(off > 0, 9.0, 6.25) + 0.2 * r.standard_normal()
            dur = np.where((doy > 150) & (doy < 270), 0.4, 0.9)  # solar water heating helps in summer
            cols["boiler"] = np.where(((h - start_h) % 24) < dur, 2000.0, 0.0)
        wm = "washer_dryer" if st["has_washer_dryer"] else ("washing_machine" if st["has_washing_machine"] else None)
        if wm:
            cols[wm] = _cycle(h - _events(days, r, 0.4, 17.5, 21.0), [(0.25, 2000.0), (1.25, 160.0)])
        if st["has_dishwasher"]:
            cols["dishwasher"] = _cycle(h - _events(days, r, 0.8, 21.0, 22.5), [(0.5, 1800.0), (1.0, 90.0)])
        if st["has_kettle"]:
            cols["kettle"] = np.where(np.abs(h - (wake + 0.3)) < 0.06, 2000.0, 0.0)
        cols["fridge"] = np.where((np.arange(len(h)) * step_s // 1200) % 2 == 0, 110.0, 8.0) * st["n_fridge"]
        unmetered = 90 + 35 * occ + routine + _cycle(h - _events(days, r, 0.6, 12.5, 14.5), [(0.75, 1500.0)])
        if house in (3, 10):  # family away in August
            away = ((days >= "2023-08-07") & (days <= "2023-08-20")).to_numpy()
            for k in cols:
                if k != "fridge":
                    cols[k] = np.where(away, 0.0, cols[k])
            unmetered = np.where(away, 60.0, unmetered)
        apps = pd.DataFrame(cols)
        p_agg = (apps.sum(axis=1).to_numpy() + unmetered) * r.lognormal(0, 0.12, len(h))
        missing = np.zeros(len(h), dtype=bool)
        for _ in range(25):
            i0 = int(r.integers(0, len(h)))
            missing[i0:i0 + int(r.integers(3, 6 * 3600 // step_s))] = True
        if house == 5:  # mains switched off during an absence
            missing |= ((days >= "2023-08-05") & (days <= "2023-08-20")).to_numpy()
        volts = 230 + r.normal(0, 2, len(h))
        elec = pd.DataFrame({"timestamp": stamp, "V": volts, "A": p_agg / volts, "P_agg": p_agg})
        for k in apps.columns:
            elec[k] = apps[k].to_numpy()
        elec["issues"] = (apps.sum(axis=1).to_numpy() > p_agg).astype(int)
        elec.loc[missing, ["V", "A", "P_agg", *apps.columns]] = np.nan
        for mth, g in elec.groupby(month):
            g.round(2).to_csv(hd / "Electric_data" / f"{mth}.csv", index=False)
        pd.DataFrame({
            "appliance": list(apps.columns),
            "wattage [W]": [2500 if c.startswith(("boiler", "kettle")) else 2200 for c in apps.columns],
            "threshold [W]": [15 if c == "fridge" else 50 for c in apps.columns],
            "min_on (sec)": [60] * apps.shape[1],
            "min_off (sec)": [60] * apps.shape[1],
        }).to_csv(hd / "Electric_data" / "appliances_metadata.csv", index=False)
        int_temp = np.clip(22 + 0.3 * (temp - 19) - 0.002 * ac_total * (temp > 26), 14, 33)
        env = pd.DataFrame({"timestamp": stamp, "internal_temperature": int_temp,
                            "internal_humidity": np.clip(hum - 8, 20, 80),
                            "external_temperature": temp, "external_humidity": hum})[env_mask]
        for mth, g in env.groupby(month[env_mask]):
            g.round(2).to_csv(hd / "Environmental_data" / f"{mth}.csv", index=False)
        (hd / f"metadata_H{house}.txt").write_text(
            f"House {house}\nMonitored appliances: {', '.join(apps.columns)}\nRecorded values: V, A, P_agg\n",
            encoding="utf-8")
    return base_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--houses", type=int, default=13)
    ap.add_argument("--start", default="2022-07-15")
    ap.add_argument("--end", default="2023-10-01")
    ap.add_argument("--step", type=int, default=300, help="sampling step in seconds (the real data use 10)")
    ap.add_argument("--clock", default="fixed_utc3", choices=["fixed_utc3", "local"])
    a = ap.parse_args()
    print(make(a.out, a.houses, a.start, a.end, a.step, a.clock))
