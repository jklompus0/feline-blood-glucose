"""Feature engineering + chronological split.

Every history feature is built from *earlier* cycles only (shift(1)), so
nothing from the cycle being predicted leaks into its features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOW_BG = 80.0  # mg/dL — binary target: reading below this is "low"
HOUR_GRID = np.arange(1, 12, dtype=float)  # for the sequence model
CATS = ["Cat_2", "Cat_3", "Cat_4"]

SHOT_FEATURES = [
    "hours", "log_pre", "pre_missing", "dose", "dose_missing", "dose_per_100",
    "is_pm", "no_shot", "partial", "gap_h", "dose_change",
    "prev_min", "prev_min_missing", "prev_pre", "roll_min4", "roll_pre4",
    "roll_low_frac8", "days_in",
]
INCYCLE_FEATURES = ["last_log_bg", "last_hours", "since_last", "slope_per_h", "has_last"]


def cycle_features(cycles: pd.DataFrame, readings: pd.DataFrame) -> pd.DataFrame:
    c = cycles.sort_values(["cat", "t0"]).copy()
    c["t0"] = pd.to_datetime(c["t0"])
    cyc_min = readings.groupby("cycle")["bg"].min()
    cyc_low = readings.assign(low=readings.bg < LOW_BG).groupby("cycle")["low"].mean()
    c["obs_min"] = c["cycle"].map(cyc_min)
    c["obs_low_frac"] = c["cycle"].map(cyc_low)

    c["pre_missing"] = c.pre_bg.isna().astype(float)
    c["dose_missing"] = c.dose.isna().astype(float)
    g = c.groupby("cat")
    # forward-fill missing pre/dose from the previous cycle of the same cat
    c["pre_f"] = g.pre_bg.ffill()
    c["dose"] = g.dose.ffill().fillna(0.0)
    c["log_pre"] = np.log(c.pre_f.fillna(c.pre_f.median()))
    c["dose_per_100"] = c.dose / np.exp(c.log_pre) * 100
    c["is_pm"] = (c.half == "pm").astype(float)
    c["no_shot"] = c.no_shot.astype(float)
    c["partial"] = c.partial.astype(float)

    g = c.groupby("cat")
    c["gap_h"] = g.t0.diff().dt.total_seconds().div(3600).fillna(12).clip(0, 72)
    c["dose_change"] = g.dose.diff().fillna(0.0)
    c["prev_min"] = g.obs_min.shift(1)
    c["prev_min_missing"] = c.prev_min.isna().astype(float)
    c["prev_pre"] = g.pre_f.shift(1)
    c["roll_min4"] = g.obs_min.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    c["roll_pre4"] = g.pre_f.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    c["roll_low_frac8"] = g.obs_low_frac.transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).mean())
    c["days_in"] = (c.t0 - g.t0.transform("min")).dt.total_seconds() / 86400

    for col in ["prev_min", "prev_pre", "roll_min4", "roll_pre4"]:
        c[col] = np.log(c[col].fillna(c.pre_f).fillna(c.pre_f.median()))
    c["roll_low_frac8"] = c.roll_low_frac8.fillna(0.0)
    return c


def reading_table(cycles_f: pd.DataFrame, readings: pd.DataFrame) -> pd.DataFrame:
    """One row per post-shot reading, with shot-time + in-cycle features."""
    r = readings.drop(columns=["t0", "half"]).merge(cycles_f, on=["cat", "cycle"])
    r = r.sort_values(["cycle", "hours"]).reset_index(drop=True)
    r["y"] = (r.bg < LOW_BG).astype(int)

    # in-cycle: most recent *earlier* reading in the same cycle (pre-shot BG at h=0
    # if none) — what an owner would know when deciding whether to test again.
    g = r.groupby("cycle")
    last_bg = g.bg.shift(1)
    last_h = g.hours.shift(1)
    r["has_last"] = last_bg.notna().astype(float)
    pre = np.exp(r.log_pre)
    r["last_log_bg"] = np.log(last_bg.fillna(pre))
    r["last_hours"] = last_h.fillna(0.0)
    r["since_last"] = r.hours - r.last_hours
    r["slope_per_h"] = np.where(
        r.last_hours > 0, (r.last_log_bg - r.log_pre) / r.last_hours.replace(0, np.nan), 0.0)
    r["slope_per_h"] = r.slope_per_h.fillna(0.0)
    return r


def chrono_split(cycles_f: pd.DataFrame, train=0.70, val=0.15) -> dict[str, str]:
    """Per cat, oldest 70% of cycles -> train, next 15% -> val, newest 15% -> test."""
    out = {}
    for _, grp in cycles_f.sort_values("t0").groupby("cat"):
        n = len(grp)
        a, b = int(n * train), int(n * (train + val))
        for i, cid in enumerate(grp.cycle):
            out[cid] = "train" if i < a else "val" if i < b else "test"
    return out


def history_tensor(cycles_f: pd.DataFrame, readings: pd.DataFrame, k: int = 6):
    """For each cycle, the previous k cycles of the same cat as a sequence of
    [log_pre, dose, is_pm, BG at +1..+11 (log, 0 if missing), mask x11]."""
    piv = (readings.assign(h=readings.hours.round().clip(1, 11))
           .groupby(["cycle", "h"]).bg.min().unstack())
    piv = piv.reindex(columns=HOUR_GRID)
    step_dim = 3 + 2 * len(HOUR_GRID)
    hist = {}
    for _, grp in cycles_f.sort_values("t0").groupby("cat"):
        rows = []
        for _, c in grp.iterrows():
            bg = piv.loc[c.cycle].values if c.cycle in piv.index else np.full(11, np.nan)
            mask = ~np.isnan(bg)
            vec = np.concatenate([[c.log_pre - 5.0, c.dose, c.is_pm],
                                  np.where(mask, np.log(np.where(mask, bg, 1)) - 5.0, 0.0),
                                  mask.astype(float)])
            rows.append(vec)
        rows = np.array(rows, dtype=np.float32)
        for i, cid in enumerate(grp.cycle):
            seq = np.zeros((k, step_dim), dtype=np.float32)
            past = rows[max(0, i - k):i]
            if len(past):
                seq[-len(past):] = past
            hist[cid] = seq
    return hist
