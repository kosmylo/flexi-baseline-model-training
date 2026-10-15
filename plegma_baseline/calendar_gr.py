"""Greek calendar features: public holidays, Orthodox Easter week, vacation periods, reduced-tariff windows."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from dateutil.easter import EASTER_ORTHODOX, easter


def orthodox_easter(year: int) -> pd.Timestamp:
    return pd.Timestamp(easter(int(year), EASTER_ORTHODOX))


def greek_holidays(years: Iterable[int]) -> list[pd.Timestamp]:
    """National public holidays. Uses the `holidays` package, with a built-in fallback list."""
    years = sorted({int(y) for y in years})
    try:
        import holidays

        days = {pd.Timestamp(d) for d in holidays.country_holidays("GR", years=years)}
    except Exception:
        days = set()
        for y in years:
            for md in ("01-01", "01-06", "03-25", "05-01", "08-15", "10-28", "12-25", "12-26"):
                days.add(pd.Timestamp(f"{y}-{md}"))
            e = orthodox_easter(y)
            for k in (-48, -2, 0, 1, 50):  # Clean Monday, Good Friday, Easter, Easter Monday, Whit Monday
                days.add(e + pd.Timedelta(days=k))
    return sorted(pd.Timestamp(d).as_unit("ns") for d in days)


def is_day_off(dates: pd.Series, hol: list[pd.Timestamp]) -> np.ndarray:
    """Weekend or public holiday."""
    return ((dates.dt.dayofweek >= 5) | dates.isin(hol)).to_numpy()


def in_periods(dates: pd.Series, periods) -> np.ndarray:
    """Membership in MM-DD ranges. A range may wrap the year end."""
    md = (dates.dt.month * 100 + dates.dt.day).to_numpy()
    out = np.zeros(len(dates), dtype=bool)
    for start, end in periods or []:
        s, e = int(start.replace("-", "")), int(end.replace("-", ""))
        out |= ((md >= s) & (md <= e)) if s <= e else ((md >= s) | (md <= e))
    return out


def easter_week(dates: pd.Series) -> np.ndarray:
    """Holy Week through Easter Monday, Orthodox calendar."""
    out = np.zeros(len(dates), dtype=bool)
    for y in pd.unique(dates.dt.year):
        e = orthodox_easter(y)
        out |= ((dates >= e - pd.Timedelta(days=6)) & (dates <= e + pd.Timedelta(days=1))).to_numpy()
    return out


def _in_window(hours: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < end:
        return (hours >= start) & (hours < end)
    return (hours >= start) | (hours < end)


def tariff_flag(local: pd.Series, regimes) -> np.ndarray:
    """True when the local hour falls in a reduced-rate window of the regime valid on that date."""
    hours = local.dt.hour.to_numpy()
    months = local.dt.month.to_numpy()
    dates = local.dt.normalize()
    out = np.zeros(len(local), dtype=bool)
    for reg in regimes or []:
        active = np.ones(len(local), dtype=bool)
        if reg.get("valid_from"):
            active &= (dates >= pd.Timestamp(reg["valid_from"])).to_numpy()
        if reg.get("valid_until"):
            active &= (dates <= pd.Timestamp(reg["valid_until"])).to_numpy()
        for season in reg["seasons"]:
            in_season = np.isin(months, season["months"])
            for start, end in season["windows"]:
                out |= active & in_season & _in_window(hours, int(start), int(end))
    return out
