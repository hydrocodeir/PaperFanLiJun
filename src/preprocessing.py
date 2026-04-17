from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_CANONICAL_COLUMNS = [
    "station_id",
    "station_name",
    "year",
    "month",
    "day",
    "tmin",
    "tmax",
]
OPTIONAL_CANONICAL_COLUMNS = ["precip"]


@dataclass
class PreprocessingArtifacts:
    daily: pd.DataFrame
    quality_summary: pd.DataFrame
    issue_log: pd.DataFrame


def load_configured_csv(config: Dict) -> pd.DataFrame:
    data_cfg = config["data"]
    path = Path(data_cfg["input_csv"])
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. Place your CSV there before running the pipeline."
        )

    read_kwargs = data_cfg.get("read_csv_kwargs", {}) or {}
    df = pd.read_csv(path, **read_kwargs)

    column_map = data_cfg.get("column_map", {}) or {}
    df = df.rename(columns=column_map)

    missing_required = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            "Missing required columns after applying column_map: "
            + ", ".join(missing_required)
        )

    if "tmean" not in df.columns:
        df["tmean"] = (pd.to_numeric(df["tmin"], errors="coerce") + pd.to_numeric(df["tmax"], errors="coerce")) / 2

    keep_cols = list(dict.fromkeys(REQUIRED_CANONICAL_COLUMNS + ["tmean"]))
    for col in OPTIONAL_CANONICAL_COLUMNS:
        if col in df.columns:
            keep_cols.append(col)
    df = df[keep_cols].copy()
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["station_id"] = out["station_id"].astype(str)
    out["station_name"] = out["station_name"].astype(str)

    for c in ["year", "month", "day"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    for c in ["tmin", "tmean", "tmax", "precip"]:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["date"] = pd.to_datetime(
        dict(year=out["year"], month=out["month"], day=out["day"]),
        errors="coerce",
    )
    return out


def _drop_invalid_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    invalid = df[df["date"].isna()].copy()
    valid = df[df["date"].notna()].copy()
    if not invalid.empty:
        invalid["issue"] = "invalid_date"
    return valid, invalid


def _apply_physical_checks(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    qc = config["preprocessing"]["qc"]
    out = df.copy()
    issues: List[pd.DataFrame] = []

    for var in ["tmin", "tmean", "tmax"]:
        lo, hi = qc["temperature_bounds_celsius"][var]
        mask = out[var].notna() & ((out[var] < lo) | (out[var] > hi))
        if mask.any():
            issue = out.loc[mask, ["station_id", "station_name", "date", var]].copy()
            issue["issue"] = f"{var}_outside_bounds"
            issues.append(issue)
            out.loc[mask, var] = np.nan

    swap_mask = out["tmin"].notna() & out["tmax"].notna() & (out["tmin"] > out["tmax"])
    if swap_mask.any():
        issue = out.loc[swap_mask, ["station_id", "station_name", "date", "tmin", "tmax"]].copy()
        issue["issue"] = "tmin_gt_tmax_swapped"
        issues.append(issue)
        tmp = out.loc[swap_mask, "tmin"].copy()
        out.loc[swap_mask, "tmin"] = out.loc[swap_mask, "tmax"].values
        out.loc[swap_mask, "tmax"] = tmp.values

    bad_mean = (
        out["tmean"].notna()
        & out["tmin"].notna()
        & out["tmax"].notna()
        & ((out["tmean"] < out["tmin"]) | (out["tmean"] > out["tmax"]))
    )
    if bad_mean.any():
        issue = out.loc[bad_mean, ["station_id", "station_name", "date", "tmin", "tmean", "tmax"]].copy()
        issue["issue"] = "tmean_outside_tmin_tmax_recomputed"
        issues.append(issue)
    out["tmean"] = np.where(
        out["tmin"].notna() & out["tmax"].notna(),
        (out["tmin"] + out["tmax"]) / 2,
        out["tmean"],
    )

    issue_log = pd.concat(issues, ignore_index=True) if issues else pd.DataFrame()
    return out, issue_log


def _deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key = ["station_id", "date"]
    dup_mask = df.duplicated(subset=key, keep=False)
    dups = df.loc[dup_mask].sort_values(key).copy()
    if not dups.empty:
        dups["issue"] = "duplicate_station_date_keep_last"
    out = df.sort_values(key).drop_duplicates(subset=key, keep="last").copy()
    return out, dups


def _add_time_fields(df: pd.DataFrame, leap_day_strategy: str) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["is_leap_day"] = (out["month"] == 2) & (out["day"] == 29)

    if leap_day_strategy == "drop":
        out = out.loc[~out["is_leap_day"]].copy()

    out["doy"] = out["date"].dt.dayofyear
    if leap_day_strategy == "drop":
        after_feb28_nonleap = (
            (out["date"].dt.is_leap_year)
            & (out["date"].dt.month > 2)
        )
        out.loc[after_feb28_nonleap, "doy"] = out.loc[after_feb28_nonleap, "doy"] - 1
        out["days_in_year"] = np.where(out["date"].dt.is_leap_year, 365, 365)
    else:
        out["days_in_year"] = np.where(out["date"].dt.is_leap_year, 366, 365)

    return out


def _expected_station_calendar(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_id, g in df.groupby("station_id"):
        name = g["station_name"].iloc[0]
        all_days = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        existing = pd.DataFrame({"date": all_days})
        existing["station_id"] = station_id
        existing["station_name"] = name
        rows.append(existing)
    return pd.concat(rows, ignore_index=True)


def _summarize_quality(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["station_id", "station_name", "year"], dropna=False)
    out = grp.agg(
        n_rows=("date", "size"),
        valid_tmax=("tmax", lambda s: s.notna().sum()),
        valid_tmin=("tmin", lambda s: s.notna().sum()),
        valid_tmean=("tmean", lambda s: s.notna().sum()),
        first_date=("date", "min"),
        last_date=("date", "max"),
    ).reset_index()

    expected = np.where(
        pd.to_datetime(out["year"].astype(int).astype(str) + "-01-01").dt.is_leap_year,
        366,
        365,
    )
    out["expected_days"] = expected
    out["tmax_coverage"] = out["valid_tmax"] / out["expected_days"]
    out["tmin_coverage"] = out["valid_tmin"] / out["expected_days"]
    out["tmean_coverage"] = out["valid_tmean"] / out["expected_days"]
    return out.sort_values(["station_id", "year"]).reset_index(drop=True)


def preprocess_temperature_data(config: Dict) -> PreprocessingArtifacts:
    df = load_configured_csv(config)
    df = _coerce_types(df)
    valid_dates, invalid_dates = _drop_invalid_dates(df)

    deduped, duplicates = _deduplicate(valid_dates)
    checked, qc_issues = _apply_physical_checks(deduped, config)
    checked = _add_time_fields(checked, config["preprocessing"]["leap_day_strategy"])

    issue_frames = [
        x for x in [invalid_dates, duplicates, qc_issues] if x is not None and not x.empty
    ]
    issue_log = pd.concat(issue_frames, ignore_index=True) if issue_frames else pd.DataFrame()

    quality_summary = _summarize_quality(checked)
    return PreprocessingArtifacts(
        daily=checked.sort_values(["station_id", "date"]).reset_index(drop=True),
        quality_summary=quality_summary,
        issue_log=issue_log,
    )
