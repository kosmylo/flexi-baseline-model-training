"""Turn the Plegma clean release into hourly, analysis-ready tables.

Outputs in processed_dir:
  hourly_load.parquet        house, utc, load_kwh, mean_w, coverage, valid, issues_share
  hourly_appliances.parquet  house, utc, appliance, kwh, coverage
  hourly_env.parquet         house, utc, int_temp, int_hum, ext_temp, ext_hum
  prepare_report.json        clock decision, per-house coverage and parsing notes
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TIME_NAMES = {"timestamp", "datetime", "date_time", "time", "date"}
TIME_FORMATS = (
    "%m/%d/%Y %I:%M:%S %p",  # format given in the data descriptor
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)
NON_APPLIANCE = {"v", "a", "voltage", "current", "issues"}
ENV_NAMES = (("int", "temp", "int_temp"), ("int", "hum", "int_hum"),
             ("ext", "temp", "ext_temp"), ("ext", "hum", "ext_hum"))
# Shift of the daily profile on the raw clock, in hours, if timestamps keep UTC+3 all year
DST_CHECKS = (("2022-10-30", 1), ("2023-03-26", -1), ("2023-10-29", 1), ("2024-03-31", -1))


def _subdir(parent: Path, name: str) -> Path | None:
    for p in parent.iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return None


def find_house_dirs(root: Path) -> dict[int, Path]:
    """House number to clean-data folder (a House_<i> folder that holds Electric_data)."""
    houses: dict[int, Path] = {}
    for d in sorted(Path(root).rglob("*")):
        if d.is_dir() and re.fullmatch(r"house[_\- ]?0*\d+", d.name, flags=re.IGNORECASE) and _subdir(d, "Electric_data"):
            houses.setdefault(int(re.search(r"\d+", d.name).group()), d)
    return dict(sorted(houses.items()))


def parse_times(values: pd.Series) -> tuple[pd.Series, str]:
    """Parse time strings to naive timestamps on the file's own clock."""
    s = values.astype(str).str.strip()
    best, best_ok, best_fmt = None, -1.0, ""
    for fmt in TIME_FORMATS:
        t = pd.to_datetime(s, format=fmt, errors="coerce")
        ok = float(t.notna().mean()) if len(t) else 0.0
        if ok > best_ok:
            best, best_ok, best_fmt = t, ok, fmt
        if ok >= 0.999:
            break
    if best_ok < 0.99:
        if s.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).mean() > 0.5:
            # explicit offsets: express them on the Athens wall clock
            t = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
            t = t.dt.tz_convert("Europe/Athens").dt.tz_localize(None)
            tag = "explicit-offset"
        else:
            t = pd.to_datetime(s, errors="coerce", format="mixed")
            tag = "mixed"
        if float(t.notna().mean()) > best_ok:
            best, best_fmt = t, tag
    return best.dt.as_unit("ns"), best_fmt


def _step_seconds(t: pd.Series) -> float:
    u = np.unique(t.dropna().to_numpy())
    if len(u) < 3:
        return float("nan")
    d = np.diff(u).astype("timedelta64[ms]").astype(np.int64) / 1000.0
    d = d[d > 0]
    return float(np.median(d)) if len(d) else float("nan")


def _read(path: Path) -> tuple[pd.DataFrame, pd.Series, str]:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    tcol = next((c for c in df.columns if c.lower() in TIME_NAMES), df.columns[0])
    t, fmt = parse_times(df[tcol])
    keep = t.notna().to_numpy()
    num = df.loc[keep].drop(columns=[tcol]).apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    return num, t[keep].reset_index(drop=True), fmt


def _hourly(t: pd.Series, values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = values.groupby(t.dt.floor("h").to_numpy())
    return g.sum(min_count=1), g.count()


def read_electric(path: Path):
    """Hourly sums and sample counts of P_agg, issues and every appliance column in one monthly file."""
    num, t, fmt = _read(path)
    low = {c: c.lower() for c in num.columns}
    pagg = [c for c in num.columns if low[c] == "p_agg"] or [c for c in num.columns if low[c].startswith("p_agg")]
    if not pagg:
        raise ValueError(f"no P_agg column among {list(num.columns)}")
    out = pd.DataFrame({"P_agg": num[pagg].sum(axis=1, min_count=1)})  # several P_agg columns are summed (phases)
    issues = [c for c in num.columns if low[c] == "issues"]
    if issues:
        out["issues"] = num[issues[0]]
    for c in num.columns:
        if low[c] in NON_APPLIANCE or low[c].startswith(("p_agg", "unnamed")):
            continue
        out["app:" + c] = num[c]
    sums, counts = _hourly(t, out)
    return sums, counts, _step_seconds(t), fmt


def read_env(path: Path):
    num, t, fmt = _read(path)
    rename: dict[str, str] = {}
    for c in num.columns:
        low = c.lower()
        for a, b, name in ENV_NAMES:
            if a in low and b in low and name not in rename.values():
                rename[c] = name
                break
    if not rename:
        return None
    sums, counts = _hourly(t, num[list(rename)].rename(columns=rename))
    return sums, counts, fmt


def _combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(parts).groupby(level=0).sum(min_count=1)


def read_house(house: int, hdir: Path) -> dict:
    edir = _subdir(hdir, "Electric_data")
    s_parts, c_parts, steps, fmts, files = [], [], [], set(), []
    for f in sorted(edir.glob("*.csv")):
        if "metadata" in f.name.lower():
            continue
        try:
            sums, counts, step, fmt = read_electric(f)
        except Exception as exc:
            log.warning("house %s: skipping %s (%s)", house, f.name, exc)
            continue
        s_parts.append(sums)
        c_parts.append(counts)
        steps.append(step)
        fmts.add(fmt)
        files.append(f.name)
    if not s_parts:
        raise ValueError(f"house {house}: no readable electric files in {edir}")
    env = None
    vdir = _subdir(hdir, "Environmental_data")
    if vdir:
        e_s, e_c = [], []
        for f in sorted(vdir.glob("*.csv")):
            try:
                r = read_env(f)
            except Exception as exc:
                log.warning("house %s: skipping %s (%s)", house, f.name, exc)
                continue
            if r:
                e_s.append(r[0])
                e_c.append(r[1])
        if e_s:
            ec = _combine(e_c).fillna(0)
            env = _combine(e_s) / ec.where(ec > 0)
    meta_file = edir / "appliances_metadata.csv"
    return {
        "house": house, "sums": _combine(s_parts), "counts": _combine(c_parts).fillna(0),
        "step_s": float(np.nanmedian(steps)), "formats": sorted(fmts), "files": files, "env": env,
        "appliance_meta": pd.read_csv(meta_file).to_dict("records") if meta_file.exists() else [],
    }


def _raw_load(h: dict) -> pd.DataFrame:
    n = h["counts"]["P_agg"]
    return pd.DataFrame({
        "house": h["house"], "hour_raw": h["sums"].index,
        "mean_w": (h["sums"]["P_agg"] / n.where(n > 0)).to_numpy(),
        "coverage": (n / (3600.0 / h["step_s"])).to_numpy(),
    })


def _profile(d: pd.DataFrame, min_days: int):
    if d["hour_raw"].dt.normalize().nunique() < min_days:
        return None
    prof = d.groupby(d["hour_raw"].dt.hour)["mean_w"].mean().reindex(range(24)).interpolate(limit_direction="both")
    if prof.isna().any() or prof.mean() <= 0:
        return None
    return (prof / prof.mean()).to_numpy()


def infer_clock(raw: pd.DataFrame, min_days: int = 7) -> dict:
    """Tell a fixed UTC+3 clock from a local wall clock with daily profiles around DST changes.

    Routines follow local time. On a fixed UTC+3 clock the daily profile moves by one hour
    at each DST change. On a local clock it stays put.
    """
    d_local = d_fixed = 0.0
    n = 0
    for day, shift in DST_CHECKS:
        t0 = pd.Timestamp(day)
        before = raw[(raw["hour_raw"] >= t0 - pd.Timedelta(days=14)) & (raw["hour_raw"] < t0)]
        after = raw[(raw["hour_raw"] >= t0 + pd.Timedelta(days=1)) & (raw["hour_raw"] < t0 + pd.Timedelta(days=15))]
        for house in sorted(set(before["house"]) & set(after["house"])):
            pb = _profile(before[before["house"] == house], min_days)
            pa = _profile(after[after["house"] == house], min_days)
            if pb is None or pa is None:
                continue
            d_local += float(np.abs(pb - pa).mean())
            d_fixed += float(np.abs(pb - np.roll(pa, -shift)).mean())
            n += 1
    if n == 0:
        return {"decision": None, "reason": "no house has 7 days of data on both sides of a DST change"}
    ratio = d_fixed / d_local if d_local > 0 else float("inf")
    decision = "fixed_utc3" if ratio < 0.9 else "local" if ratio > 1 / 0.9 else None
    return {"decision": decision, "fixed_vs_local_distance_ratio": round(ratio, 3), "comparisons": n}


def to_utc(hours_raw, mode: str, tz: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(hours_raw).as_unit("ns")
    if mode == "fixed_utc3":
        return (idx - pd.Timedelta(hours=3)).tz_localize("UTC")
    if mode == "utc":
        return idx.tz_localize("UTC")
    if mode == "local":
        loc = idx.tz_localize(tz, ambiguous=np.ones(len(idx), dtype=bool), nonexistent="shift_forward")
        return loc.tz_convert("UTC")
    raise ValueError(f"unknown clock mode {mode!r}")


def _utc_frame(df: pd.DataFrame, mode: str, tz: str, how: str = "sum") -> pd.DataFrame:
    df = df.copy()
    df.index = to_utc(df.index, mode, tz)
    g = df.groupby(level=0)
    return g.sum(min_count=1) if how == "sum" else g.mean()


def build_tables(houses: list[dict], mode: str, cfg: dict):
    pc, tz = cfg["prepare"], cfg["prepare"]["timezone"]
    loads, apps, envs, report = [], [], [], {}
    for h in houses:
        s = _utc_frame(h["sums"], mode, tz)
        c = _utc_frame(h["counts"], mode, tz).fillna(0)
        expected = 3600.0 / h["step_s"]
        full = pd.date_range(s.index.min(), s.index.max(), freq="h").as_unit("ns")
        s, c = s.reindex(full), c.reindex(full).fillna(0)
        n = c["P_agg"]
        mean_w = s["P_agg"] / n.where(n > 0)
        cov = (n / expected).clip(upper=1.0)
        valid = (cov >= pc["min_coverage"]) & (mean_w >= pc["min_load_w"])
        load = pd.DataFrame({
            "house": h["house"], "utc": full, "load_kwh": (mean_w / 1000.0).where(valid).to_numpy(),
            "mean_w": mean_w.to_numpy(), "coverage": cov.to_numpy(), "valid": valid.to_numpy(),
        })
        if "issues" in s.columns:
            load["issues_share"] = (s["issues"] / c["issues"].where(c["issues"] > 0)).to_numpy()
        loads.append(load)
        app_cols = [col for col in s.columns if col.startswith("app:")]
        for col in app_cols:
            nn = c[col]
            apps.append(pd.DataFrame({
                "house": h["house"], "utc": full, "appliance": col[4:],
                "kwh": (s[col] / nn.where(nn > 0) / 1000.0).to_numpy(),
                "coverage": (nn / expected).clip(upper=1.0).to_numpy(),
            }))
        if h["env"] is not None:
            e = _utc_frame(h["env"], mode, tz, how="mean").reset_index(names="utc")
            e.insert(0, "house", h["house"])
            envs.append(e)
        report[str(h["house"])] = {
            "files": len(h["files"]), "time_formats": h["formats"], "sample_step_s": h["step_s"],
            "first_utc": str(full.min()), "last_utc": str(full.max()), "hours": int(len(full)),
            "valid_hours": int(valid.sum()), "valid_share": round(float(valid.mean()), 4),
            "median_coverage": round(float(cov.median()), 4),
            "appliances": [col[4:] for col in app_cols], "appliance_metadata": h["appliance_meta"],
        }
    load = pd.concat(loads, ignore_index=True)
    app = pd.concat(apps, ignore_index=True) if apps else pd.DataFrame(columns=["house", "utc", "appliance", "kwh", "coverage"])
    env = pd.concat(envs, ignore_index=True) if envs else None
    return load, app, env, report


def prepare(cfg: dict, root: str | Path | None = None) -> dict:
    pc = cfg["prepare"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    if root is None:
        marker = raw_dir / "clean_root.txt"
        root = Path(marker.read_text(encoding="utf-8").strip()) if marker.exists() else raw_dir
    house_dirs = find_house_dirs(Path(root))
    if not house_dirs:
        raise FileNotFoundError(f"No House_<i>/Electric_data folders under {root}. Run 'download' or pass --root.")
    log.info("reading %d houses under %s", len(house_dirs), root)
    houses = []
    for n, d in house_dirs.items():
        houses.append(read_house(n, d))
        log.info("house %d: %d files, %.0f s sampling", n, len(houses[-1]["files"]), houses[-1]["step_s"])

    raw = pd.concat([_raw_load(h) for h in houses], ignore_index=True)
    raw = raw[(raw["coverage"] >= pc["min_coverage"]) & (raw["mean_w"] >= pc["min_load_w"])]
    check = infer_clock(raw)
    mode = pc["clock"]
    if mode == "auto":
        mode = check["decision"] or "fixed_utc3"
        if check["decision"] is None:
            log.warning("clock check inconclusive (%s). Using the documented fixed UTC+3 clock.", check)
    log.info("clock check %s, using '%s'", check, mode)

    load, app, env, per_house = build_tables(houses, mode, cfg)
    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    load.to_parquet(out / "hourly_load.parquet", index=False)
    app.to_parquet(out / "hourly_appliances.parquet", index=False)
    if env is not None:
        env.to_parquet(out / "hourly_env.parquet", index=False)
    report = {
        "root": str(root), "clock_mode": mode, "clock_check": check,
        "min_coverage": pc["min_coverage"], "min_load_w": pc["min_load_w"],
        "valid_share_all": round(float(load["valid"].mean()), 4), "houses": per_house,
    }
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("wrote hourly tables to %s (valid hours %.1f%%)", out, 100 * report["valid_share_all"])
    return report
