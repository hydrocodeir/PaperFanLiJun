from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class HomogenizationArtifacts:
    daily: pd.DataFrame
    breaks: pd.DataFrame
    adjustments: pd.DataFrame
    method: str


def _required_columns_ok(df: pd.DataFrame) -> bool:
    need = {"station_id", "station_name", "date", "year", "tmin", "tmax", "tmean"}
    return need.issubset(df.columns)


def _estimate_shift_celsius(
    station_df: pd.DataFrame,
    variable: str,
    break_date: pd.Timestamp,
    window_days: int = 365,
) -> float:
    before = station_df.loc[
        (station_df["date"] < break_date) & (station_df["date"] >= break_date - pd.Timedelta(days=window_days)),
        variable,
    ].dropna()
    after = station_df.loc[
        (station_df["date"] >= break_date) & (station_df["date"] < break_date + pd.Timedelta(days=window_days)),
        variable,
    ].dropna()
    if before.empty or after.empty:
        return 0.0
    return float(before.mean() - after.mean())


def _detect_breaks_mean_shift_proxy(
    daily: pd.DataFrame,
    variable: str,
    z_threshold: float = 2.5,
    min_year_gap: int = 5,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for station_id, g in daily.groupby("station_id"):
        g = g.sort_values("date")
        yearly = g.groupby("year", as_index=False)[variable].mean().dropna()
        if len(yearly) < 10:
            continue

        yearly["delta"] = yearly[variable].diff()
        sigma = float(yearly["delta"].std(ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            continue

        candidate = yearly.loc[yearly["delta"].abs() >= (z_threshold * sigma), ["year", "delta"]].copy()
        if candidate.empty:
            continue

        accepted_years: List[int] = []
        for _, row in candidate.sort_values("year").iterrows():
            year_int = int(row["year"])
            if not accepted_years or min(abs(year_int - y0) for y0 in accepted_years) >= int(min_year_gap):
                accepted_years.append(year_int)
                break_date = pd.Timestamp(year=year_int, month=1, day=1)
                rows.append(
                    {
                        "station_id": str(station_id),
                        "break_date": break_date,
                        "variable": variable,
                        "detected_by": "mean_shift_proxy",
                        "score_z": float(abs(row["delta"]) / sigma),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["station_id", "break_date", "variable", "detected_by", "score_z"])
    return pd.DataFrame(rows).sort_values(["station_id", "variable", "break_date"]).reset_index(drop=True)


def _load_external_breaks(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["station_id", "break_date", "variable", "shift_celsius", "detected_by"])
    df = pd.read_csv(path)
    expected = {"station_id", "break_date", "variable"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"External homogenization file missing columns: {', '.join(sorted(missing))}")
    out = df.copy()
    out["station_id"] = out["station_id"].astype(str)
    out["break_date"] = pd.to_datetime(out["break_date"], errors="coerce")
    out["detected_by"] = out.get("detected_by", "external_rhtests")
    if "shift_celsius" not in out.columns:
        out["shift_celsius"] = np.nan
    return out.dropna(subset=["break_date"]).reset_index(drop=True)


def _apply_break_adjustments(
    daily: pd.DataFrame,
    breaks: pd.DataFrame,
    vars_to_adjust: Tuple[str, ...] = ("tmin", "tmax", "tmean"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if breaks.empty:
        return daily.copy(), pd.DataFrame(columns=["station_id", "variable", "break_date", "shift_celsius", "n_affected_rows"])

    adjusted = daily.copy()
    adjustment_rows: List[Dict] = []

    for station_id, gb in breaks.groupby("station_id"):
        st_mask = adjusted["station_id"].astype(str) == str(station_id)
        station_df = adjusted.loc[st_mask].sort_values("date").copy()
        if station_df.empty:
            continue

        for _, br in gb.sort_values("break_date").iterrows():
            break_date = pd.Timestamp(br["break_date"])
            variable = str(br["variable"])
            if variable not in vars_to_adjust or variable not in station_df.columns:
                continue
            shift = br.get("shift_celsius", np.nan)
            if pd.isna(shift):
                shift = _estimate_shift_celsius(station_df, variable=variable, break_date=break_date, window_days=365)
            shift = float(shift)

            affected = st_mask & (adjusted["date"] < break_date) & adjusted[variable].notna()
            n_aff = int(affected.sum())
            adjusted.loc[affected, variable] = adjusted.loc[affected, variable] - shift
            adjustment_rows.append(
                {
                    "station_id": str(station_id),
                    "variable": variable,
                    "break_date": break_date,
                    "shift_celsius": shift,
                    "n_affected_rows": n_aff,
                }
            )

    if {"tmin", "tmax"}.issubset(adjusted.columns):
        adjusted["tmean"] = np.where(
            adjusted["tmin"].notna() & adjusted["tmax"].notna(),
            (adjusted["tmin"] + adjusted["tmax"]) / 2.0,
            adjusted["tmean"],
        )
    adj_df = pd.DataFrame(adjustment_rows)
    if adj_df.empty:
        adj_df = pd.DataFrame(columns=["station_id", "variable", "break_date", "shift_celsius", "n_affected_rows"])
    return adjusted, adj_df


def summarize_homogenization_breaks(breaks: pd.DataFrame) -> pd.DataFrame:
    if breaks is None or breaks.empty:
        return pd.DataFrame(columns=["station_id", "n_breaks_total", "n_breaks_tmin", "n_breaks_tmax", "n_breaks_tmean"])
    b = breaks.copy()
    b["station_id"] = b["station_id"].astype(str)
    summary = (
        b.pivot_table(index="station_id", columns="variable", values="break_date", aggfunc="count", fill_value=0)
        .rename(columns=lambda c: f"n_breaks_{c}")
        .reset_index()
    )
    for col in ["n_breaks_tmin", "n_breaks_tmax", "n_breaks_tmean"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["n_breaks_total"] = summary[["n_breaks_tmin", "n_breaks_tmax", "n_breaks_tmean"]].sum(axis=1)
    return summary.sort_values("station_id").reset_index(drop=True)


def apply_homogenization(daily: pd.DataFrame, config: Dict) -> HomogenizationArtifacts:
    if daily is None or daily.empty or not _required_columns_ok(daily):
        return HomogenizationArtifacts(
            daily=daily.copy(),
            breaks=pd.DataFrame(),
            adjustments=pd.DataFrame(),
            method="none",
        )

    hcfg = config.get("homogenization", {}) or {}
    method = str(hcfg.get("method", "none")).lower()
    vars_to_adjust = tuple(hcfg.get("variables", ["tmin", "tmax", "tmean"]))

    if method == "none":
        return HomogenizationArtifacts(
            daily=daily.copy(),
            breaks=pd.DataFrame(columns=["station_id", "break_date", "variable", "detected_by", "score_z", "shift_celsius"]),
            adjustments=pd.DataFrame(columns=["station_id", "variable", "break_date", "shift_celsius", "n_affected_rows"]),
            method=method,
        )

    if method == "external_rhtests_csv":
        ext_path = Path(hcfg.get("external_breaks_csv", "outputs/tables/rhtests_breaks.csv"))
        breaks = _load_external_breaks(ext_path)
        if "score_z" not in breaks.columns:
            breaks["score_z"] = np.nan
        adjusted, adjustments = _apply_break_adjustments(daily, breaks, vars_to_adjust=vars_to_adjust)
        return HomogenizationArtifacts(daily=adjusted, breaks=breaks, adjustments=adjustments, method=method)

    if method == "mean_shift_proxy":
        variables = [v for v in vars_to_adjust if v in daily.columns]
        all_breaks = []
        for var in variables:
            all_breaks.append(
                _detect_breaks_mean_shift_proxy(
                    daily=daily,
                    variable=var,
                    z_threshold=float(hcfg.get("z_threshold", 2.5)),
                    min_year_gap=int(hcfg.get("min_year_gap", 5)),
                )
            )
        breaks = pd.concat(all_breaks, ignore_index=True) if all_breaks else pd.DataFrame()
        if breaks.empty:
            breaks = pd.DataFrame(columns=["station_id", "break_date", "variable", "detected_by", "score_z", "shift_celsius"])
        if "shift_celsius" not in breaks.columns:
            breaks["shift_celsius"] = np.nan
        adjusted, adjustments = _apply_break_adjustments(daily, breaks, vars_to_adjust=vars_to_adjust)
        return HomogenizationArtifacts(daily=adjusted, breaks=breaks, adjustments=adjustments, method=method)

    raise ValueError(
        "Unsupported homogenization.method. Use one of: none, mean_shift_proxy, external_rhtests_csv."
    )
