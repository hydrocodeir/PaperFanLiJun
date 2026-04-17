from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


INDEX_SPECS = {
    "warm_days": {"var": "tmax", "operator": "gt", "percentile": 0.90},
    "cool_days": {"var": "tmax", "operator": "lt", "percentile": 0.10},
    "warm_nights": {"var": "tmin", "operator": "gt", "percentile": 0.90},
    "cool_nights": {"var": "tmin", "operator": "lt", "percentile": 0.10},
}


def _cyclic_interpolate_thresholds(station_df: pd.DataFrame, value_col: str) -> pd.Series:
    x = station_df["doy"].to_numpy(dtype=int)
    y = station_df[value_col].to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    valid = np.isfinite(y)
    if valid.sum() == 0:
        return pd.Series(np.nan, index=station_df.index)
    if valid.sum() == 1:
        return pd.Series(np.repeat(y[valid][0], len(y)), index=station_df.index)

    xv = x[valid]
    yv = y[valid]
    x_ext = np.concatenate([xv - 365, xv, xv + 365])
    y_ext = np.concatenate([yv, yv, yv])
    filled = np.interp(x, x_ext, y_ext)
    return pd.Series(filled[np.argsort(order)], index=station_df.index)


def compute_daily_percentile_thresholds(daily: pd.DataFrame, config: Dict) -> pd.DataFrame:
    ref_start, ref_end = config["methodology"]["reference_period"]
    ref = daily.loc[(daily["year"] >= ref_start) & (daily["year"] <= ref_end)].copy()
    if ref.empty:
        # Fallback for short sample datasets that do not cover the reference period.
        ref = daily.copy()

    min_sample = int(config["feature_engineering"]["min_reference_samples_per_doy"])
    station_ref_stats = (
        ref.groupby(["station_id", "station_name"])
        .agg(
            n_ref_years=("year", "nunique"),
            n_tmax_days=("tmax", lambda s: int(s.notna().sum())),
            n_tmin_days=("tmin", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    station_ref_stats["effective_min_sample"] = station_ref_stats["n_ref_years"].clip(lower=1).map(
        lambda n: int(min(min_sample, int(n)))
    )
    station_ref_stats = station_ref_stats[
        ["station_id", "station_name", "effective_min_sample"]
    ]
    ref = ref.merge(station_ref_stats, on=["station_id", "station_name"], how="left")
    frames = []

    for (station_id, station_name, doy), g in ref.groupby(["station_id", "station_name", "doy"]):
        row = {
            "station_id": station_id,
            "station_name": station_name,
            "doy": int(doy),
            "n_tmax": int(g["tmax"].notna().sum()),
            "n_tmin": int(g["tmin"].notna().sum()),
        }

        effective_min_sample = int(g["effective_min_sample"].iloc[0]) if "effective_min_sample" in g.columns else min_sample
        row["tmax_p10"] = g["tmax"].quantile(0.10) if row["n_tmax"] >= effective_min_sample else np.nan
        row["tmax_p90"] = g["tmax"].quantile(0.90) if row["n_tmax"] >= effective_min_sample else np.nan
        row["tmin_p10"] = g["tmin"].quantile(0.10) if row["n_tmin"] >= effective_min_sample else np.nan
        row["tmin_p90"] = g["tmin"].quantile(0.90) if row["n_tmin"] >= effective_min_sample else np.nan
        frames.append(row)

    if not frames:
        return pd.DataFrame(
            columns=[
                "doy",
                "station_id",
                "station_name",
                "n_tmax",
                "n_tmin",
                "tmax_p10",
                "tmax_p90",
                "tmin_p10",
                "tmin_p90",
            ]
        )

    thresholds = pd.DataFrame(frames)
    doy_template = pd.DataFrame({"doy": np.arange(1, 366)})

    complete_frames = []
    for (station_id, station_name), g in thresholds.groupby(["station_id", "station_name"]):
        merged = doy_template.merge(g, on="doy", how="left")
        merged["station_id"] = station_id
        merged["station_name"] = station_name

        for c in ["n_tmax", "n_tmin"]:
            merged[c] = merged[c].fillna(0).astype(int)

        for c in ["tmax_p10", "tmax_p90", "tmin_p10", "tmin_p90"]:
            merged[c] = _cyclic_interpolate_thresholds(merged, c)

        complete_frames.append(merged)

    return pd.concat(complete_frames, ignore_index=True).sort_values(["station_id", "doy"])


def apply_thresholds_and_compute_indices(
    daily: pd.DataFrame,
    thresholds: pd.DataFrame,
    config: Dict,
) -> pd.DataFrame:
    merged = daily.merge(
        thresholds,
        on=["station_id", "station_name", "doy"],
        how="left",
        validate="many_to_one",
    )

    for index_name, spec in INDEX_SPECS.items():
        value = merged[spec["var"]]
        threshold_col = f"{spec['var']}_p{int(spec['percentile'] * 100):02d}"
        threshold = merged[threshold_col]
        if spec["operator"] == "gt":
            merged[index_name] = np.where(value.notna() & threshold.notna(), value > threshold, np.nan)
        else:
            merged[index_name] = np.where(value.notna() & threshold.notna(), value < threshold, np.nan)

    min_year_coverage = config["feature_engineering"]["min_year_coverage_for_index"]
    has_precip = "precip" in merged.columns
    annual_records = []

    for (station_id, station_name, year), g in merged.groupby(["station_id", "station_name", "year"]):
        row = {
            "station_id": station_id,
            "station_name": station_name,
            "year": int(year),
            "days_in_year": int(g["days_in_year"].iloc[0]),
            "valid_tmax_days": int(g["tmax"].notna().sum()),
            "valid_tmin_days": int(g["tmin"].notna().sum()),
            "tmax_coverage": float(g["tmax"].notna().sum() / g["days_in_year"].iloc[0]),
            "tmin_coverage": float(g["tmin"].notna().sum() / g["days_in_year"].iloc[0]),
            "tmean_annual": float(g["tmean"].mean()) if g["tmean"].notna().any() else np.nan,
            "tmax_annual": float(g["tmax"].mean()) if g["tmax"].notna().any() else np.nan,
            "tmin_annual": float(g["tmin"].mean()) if g["tmin"].notna().any() else np.nan,
        }
        if has_precip:
            row["precip_annual"] = float(g["precip"].sum(skipna=True)) if g["precip"].notna().any() else np.nan

        for index_name, spec in INDEX_SPECS.items():
            source_cov = row["tmax_coverage"] if spec["var"] == "tmax" else row["tmin_coverage"]
            valid_mask = g[index_name].notna()
            observed_days = int(valid_mask.sum())
            observed_events = float(g.loc[valid_mask, index_name].sum()) if observed_days > 0 else np.nan

            row[f"{index_name}_observed"] = observed_events
            row[f"{index_name}_valid_days"] = observed_days

            if observed_days == 0 or source_cov < min_year_coverage:
                row[index_name] = np.nan
            else:
                row[index_name] = observed_events * row["days_in_year"] / observed_days

        annual_records.append(row)

    annual = pd.DataFrame(annual_records).sort_values(["station_id", "year"]).reset_index(drop=True)
    return annual


def compute_network_mean_indices(annual: pd.DataFrame, config: Dict) -> pd.DataFrame:
    method = config["methodology"]["network_aggregation"]
    index_cols = list(INDEX_SPECS.keys()) + ["tmean_annual", "tmax_annual", "tmin_annual"]
    if "precip_annual" in annual.columns:
        index_cols.append("precip_annual")

    frames = []
    for year, g in annual.groupby("year"):
        row = {"year": int(year), "n_stations": int(g["station_id"].nunique())}

        if method == "simple_mean":
            for col in index_cols:
                row[col] = float(g[col].mean()) if g[col].notna().any() else np.nan
        else:
            raise ValueError(f"Unsupported network_aggregation: {method}")

        for col in index_cols:
            row[f"{col}_count"] = int(g[col].notna().sum())

        frames.append(row)

    return pd.DataFrame(frames).sort_values("year").reset_index(drop=True)
