"""Clean the raw per-cat spreadsheets into tidy tables.

Outputs two tables:
  readings : one row per glucose reading (cat, cycle, hours since shot, BG)
  cycles   : one row per AM/PM shot cycle (pre-shot BG, dose, flags, nadir)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "Cat_CSV"
HOURS = list(range(1, 12))
NO_SHOT = re.compile(r"\b(no\s*shot|no\s*dose|ns)\b", re.I)
PARTIAL = re.compile(r"\b(fur|late|partial|half)\b", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")


def _parse_dose(raw) -> tuple[float, bool, bool]:
    """Return (dose_units, no_shot, partial_flag)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan, False, False
    s = str(raw).strip()
    no_shot = bool(NO_SHOT.search(s))
    partial = bool(PARTIAL.search(s))
    s2 = s.replace(",", ".").replace(":", ".")  # "2,5" / "2:25" typos
    m = NUM.search(s2)
    dose = float(m.group()) if m else np.nan
    if no_shot:
        dose = 0.0
    return dose, no_shot, partial


def _parse_cell(raw, base_hour: int) -> tuple[list[tuple[float, float]], bool, bool]:
    """Parse a reading cell. Handles '46 @2.5 45' (46 at +2, 45 at +2.5)
    and text notes like 'Late shot' / 'No dose'. Returns (readings, no_shot, partial)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return [], False, False
    s = str(raw).strip()
    if not s:
        return [], False, False
    no_shot = bool(NO_SHOT.search(s))
    partial = bool(PARTIAL.search(s))
    out: list[tuple[float, float]] = []
    # split on '@' markers: "46 @2.5 45" -> ["46 ", "2.5 45"]
    parts = s.split("@")
    head_nums = NUM.findall(parts[0])
    if head_nums:
        out.append((float(base_hour), float(head_nums[0])))
    for p in parts[1:]:
        nums = NUM.findall(p)
        if len(nums) >= 2:
            out.append((float(nums[0]), float(nums[1])))
    # glucose meters cap at ~600 ("HI"); drop implausible values
    out = [(h, v) for h, v in out if 10 <= v <= 750]
    return out, no_shot, partial


def load_cat(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat = path.stem
    df = pd.read_csv(path, skiprows=3, dtype=str, header=0)
    df = df.iloc[:, :28]
    df.columns = (
        ["date", "am_pre", "am_u"] + [f"am_{h}" for h in HOURS]
        + ["pm_pre", "pm_u"] + [f"pm_{h}" for h in HOURS] + ["remarks"]
    )
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    readings, cycles = [], []
    for _, row in df.iterrows():
        remark = str(row["remarks"]) if isinstance(row["remarks"], str) else ""
        for half, offset in (("am", 7), ("pm", 19)):
            dose, no_shot, partial = _parse_dose(row[f"{half}_u"])
            pre_vals, _, _ = _parse_cell(row[f"{half}_pre"], 0)
            pre = pre_vals[0][1] if pre_vals else np.nan
            pts = []
            for h in HOURS:
                vals, ns, pf = _parse_cell(row[f"{half}_{h}"], h)
                no_shot |= ns
                partial |= pf
                pts.extend(vals)
            if no_shot:
                dose = 0.0
            if np.isnan(pre) and np.isnan(dose) and not pts:
                continue
            t0 = row["date"] + pd.Timedelta(hours=offset)
            cid = f"{cat}_{row['date']:%Y%m%d}_{half}"
            cycles.append(dict(
                cat=cat, cycle=cid, t0=t0, half=half, pre_bg=pre, dose=dose,
                no_shot=no_shot, partial=partial, n_readings=len(pts),
                remark=remark,
            ))
            for h, v in pts:
                readings.append(dict(cat=cat, cycle=cid, t0=t0, half=half,
                                     hours=h, bg=v))
    return pd.DataFrame(readings), pd.DataFrame(cycles)


def load_all(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    rs, cs = zip(*(load_cat(p) for p in sorted(raw_dir.glob("*.csv"))))
    readings = pd.concat(rs, ignore_index=True)
    cycles = pd.concat(cs, ignore_index=True)
    cycles = cycles.sort_values(["cat", "t0"]).reset_index(drop=True)

    nadir = readings.groupby("cycle")["bg"].min().rename("nadir")
    cycles = cycles.merge(nadir, on="cycle", how="left")
    return readings, cycles


if __name__ == "__main__":
    r, c = load_all()
    out = Path(__file__).resolve().parent.parent / "outputs"
    out.mkdir(exist_ok=True)
    r.to_csv(out / "readings_clean.csv", index=False)
    c.to_csv(out / "cycles_clean.csv", index=False)
    print(r.groupby("cat").size(), c.groupby("cat").size(), sep="\n")
