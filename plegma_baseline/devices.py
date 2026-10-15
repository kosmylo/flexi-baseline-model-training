"""Per-house device table: power values from the dataset metadata plus measured operating power.

Reads each house's appliances_metadata.csv and the 10-second appliance columns of the clean release.
Writes to processed_dir:
  devices.csv           one row per house and device (metadata values and measured statistics)
  devices_by_house.md   one row per house, in the layout of the house overview table
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .features import load_static
from .prepare import NON_APPLIANCE, _read, _subdir, find_house_dirs

log = logging.getLogger(__name__)
DEFAULT_THRESHOLD_W = 20.0
DEFAULT_MIN_S = 60.0
ORDER = ["ac", "washing_machine", "washer_dryer", "dishwasher", "fridge", "boiler", "kettle"]
LABEL = {"ac": "AC", "washing_machine": "WM", "washer_dryer": "washer-dryer", "dishwasher": "DW",
         "fridge": "fridge", "boiler": "boiler", "kettle": "kettle"}
HEATING = {"radiator_oil": "Oil radiators", "radiator_gas": "Gas radiators", "electric_heater": "Electric heater",
           "ac": "AC", "air_to_air_heat_pump": "Air-to-air heat pump", "underfloor": "Underfloor heating"}


def device_type(name: str) -> str:
    return re.sub(r"_\d+$", "", name.strip().lower())


def read_metadata(path: Path) -> pd.DataFrame:
    """Metadata with standard names: appliance, wattage_w, cutoff_w, threshold_w, min_on_s, min_off_s."""
    cols = ["appliance", "wattage_w", "threshold_w", "min_on_s", "min_off_s"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    md = pd.read_csv(path, encoding="utf-8-sig")
    rename = {}
    for c in md.columns:
        low = str(c).strip().lower()
        if low.startswith(("appliance", "device", "name")):
            rename[c] = "appliance"
        elif "cutoff" in low:
            rename[c] = "cutoff_w"
        elif "watt" in low:
            rename[c] = "wattage_w"
        elif "thresh" in low:
            rename[c] = "threshold_w"
        elif "min_on" in low:
            rename[c] = "min_on_s"
        elif "min_off" in low:
            rename[c] = "min_off_s"
    md = md.rename(columns=rename)
    for c in cols:
        if c not in md.columns:
            md[c] = np.nan
    md["appliance"] = md["appliance"].astype(str).str.strip()
    for c in md.columns.drop("appliance"):
        md[c] = pd.to_numeric(md[c], errors="coerce")
    return md.drop_duplicates("appliance")


def _num(meta: pd.Series, key: str, default: float) -> float:
    v = meta.get(key, np.nan)
    return float(v) if pd.notna(v) else default


def _runs(on: np.ndarray, min_on: int, min_off: int) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) indices of on-runs, joining gaps up to min_off and dropping runs below min_on."""
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if len(starts) == 0:
        return starts, ends
    first = np.flatnonzero(np.concatenate([[True], (starts[1:] - ends[:-1]) > min_off]))
    last = np.concatenate([first[1:] - 1, [len(starts) - 1]])
    starts, ends = starts[first], ends[last]
    keep = (ends - starts) >= min_on
    return starts[keep], ends[keep]


def device_stats(x: np.ndarray, step: float, thr: float, min_on_s: float, min_off_s: float) -> dict:
    valid = np.isfinite(x)
    on = valid & (x >= thr)
    starts, ends = _runs(on, max(1, int(round(min_on_s / step))), int(round(min_off_s / step)))
    cum = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0), dtype=float)])
    run_kwh = (cum[ends] - cum[starts]) * step / 3.6e6
    days = valid.sum() * step / 86400.0
    return {
        "on_power_median_w": float(np.median(x[on])) if on.any() else np.nan,
        "on_power_p95_w": float(np.percentile(x[on], 95)) if on.any() else np.nan,
        "max_w": float(np.nanmax(x)) if valid.any() else np.nan,
        "runs": int(len(starts)),
        "runs_per_week": float(len(starts) / (days / 7.0)) if days > 0 else np.nan,
        "run_energy_median_kwh": float(np.median(run_kwh)) if len(starts) else np.nan,
        "run_duration_median_min": float(np.median(ends - starts) * step / 60.0) if len(starts) else np.nan,
        "energy_kwh": float(cum[-1] * step / 3.6e6),
        "observed_days": float(days),
    }


def _house_data(hdir: Path) -> tuple[pd.DataFrame, float]:
    parts = []
    for f in sorted(_subdir(hdir, "Electric_data").glob("*.csv")):
        if "metadata" in f.name.lower():
            continue
        num, t, _ = _read(f)
        cols = [c for c in num.columns if c.lower() not in NON_APPLIANCE and not c.lower().startswith("unnamed")]
        part = num[cols].astype("float32")
        part.index = pd.DatetimeIndex(t)
        parts.append(part)
    data = pd.concat(parts).sort_index()
    data = data[~data.index.duplicated()]
    gaps = np.diff(data.index.values).astype("timedelta64[ms]").astype(np.int64) / 1000.0
    step = float(np.median(gaps[gaps > 0]))
    return data.resample(f"{int(round(step))}s").mean(), step


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if np.isfinite(v) else "n/a"


def _device_text(rows: pd.DataFrame) -> str:
    parts = []
    for kind in ORDER + sorted(set(rows["type"]) - set(ORDER)):
        g = rows[rows["type"] == kind]
        if g.empty:
            continue
        watts = ", ".join(f"{_fmt(v)} W" for v in g["wattage_w"])
        count = f"{len(g)} " if len(g) > 1 else ""
        parts.append(f"{count}{LABEL.get(kind, kind)} ({watts})")
    return ", ".join(parts)


def build_device_table(cfg: dict, root: str | Path | None = None) -> pd.DataFrame:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    if root is None:
        marker = raw_dir / "clean_root.txt"
        root = Path(marker.read_text(encoding="utf-8").strip()) if marker.exists() else raw_dir
    houses = find_house_dirs(Path(root))
    if not houses:
        raise FileNotFoundError(f"No House_<i>/Electric_data folders under {root}")
    rows, summary = [], []
    for house, hdir in houses.items():
        data, step = _house_data(hdir)
        md = read_metadata(_subdir(hdir, "Electric_data") / "appliances_metadata.csv").set_index("appliance")
        pagg = [c for c in data.columns if c.lower().startswith("p_agg")]
        total = data[pagg].sum(axis=1, min_count=1).to_numpy(dtype=float) if pagg else None
        agg_kwh = float(np.nansum(total) * step / 3.6e6) if total is not None else np.nan
        dev_kwh = 0.0
        for col in [c for c in data.columns if c not in pagg]:
            meta = md.loc[col] if col in md.index else pd.Series(dtype=float)
            stats = device_stats(data[col].to_numpy(dtype=float), step,
                                 _num(meta, "threshold_w", DEFAULT_THRESHOLD_W),
                                 _num(meta, "min_on_s", DEFAULT_MIN_S), _num(meta, "min_off_s", DEFAULT_MIN_S))
            dev_kwh += stats["energy_kwh"]
            rows.append({"house": house, "device": col, "type": device_type(col),
                         **{k: meta.get(k, np.nan) for k in ("wattage_w", "cutoff_w", "threshold_w", "min_on_s", "min_off_s")},
                         **stats})
        summary.append({"house": house, "aggregate_kwh": agg_kwh,
                        "submetered_share": dev_kwh / agg_kwh if agg_kwh > 0 else np.nan})
        log.info("house %d: %d devices, %.0f s sampling", house, len([r for r in rows if r["house"] == house]), step)
    table = pd.DataFrame(rows)
    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "devices.csv", index=False, float_format="%.3f")
    _write_markdown(table, pd.DataFrame(summary).set_index("house"), out / "devices_by_house.md")
    log.info("wrote %s and %s", out / "devices.csv", out / "devices_by_house.md")
    return table


def _write_markdown(table: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    static = load_static().set_index("house")
    lines = ["| House | Occupants | Dwelling | Main heating | Monitored devices (dataset wattage) | Aggregate kWh | Sub-metered |",
             "|---|---|---|---|---|---|---|"]
    for house, g in table.groupby("house"):
        st = static.loc[house] if house in static.index else None
        occ = str(int(st["occupants"])) if st is not None else ""
        dwelling = (f"{str(st['dwelling']).capitalize()}, {int(st['built_year'])}, {int(st['rooms'])} rooms"
                    + (", 3-phase" if int(st["three_phase"]) else "")) if st is not None else ""
        heating = HEATING.get(str(st["heating"]), str(st["heating"])) if st is not None else ""
        s = summary.loc[house]
        share = f"{100 * s['submetered_share']:.0f}%" if np.isfinite(s["submetered_share"]) else "n/a"
        lines.append(f"| {house} | {occ} | {dwelling} | {heating} | {_device_text(g)} | {_fmt(s['aggregate_kwh'])} | {share} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
