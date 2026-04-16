from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import Rbf, griddata
from shapely import contains_xy

SERIES_ORDER = ["warm_days", "cool_days", "warm_nights", "cool_nights"]
SERIES_PANEL_LABELS = {
    "warm_days": "(a) Warm days",
    "cool_days": "(b) Cool days",
    "warm_nights": "(c) Warm nights",
    "cool_nights": "(d) Cool nights",
}


def _series_label(series: str) -> str:
    return series.replace("_", " ").title()


def _load_boundary_geometry(boundary_path: Optional[Path]):
    if boundary_path is None:
        return None
    path = Path(boundary_path)
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    if gdf.empty:
        return None
    geom = gdf.to_crs(4326).geometry.union_all() if hasattr(gdf, "to_crs") else gdf.geometry.union_all()
    return geom


def _get_plot_extent(meta: pd.DataFrame, boundary_geom=None, pad_deg: float = 1.0):
    if boundary_geom is not None:
        minx, miny, maxx, maxy = boundary_geom.bounds
    else:
        minx = float(meta["longitude"].min())
        maxx = float(meta["longitude"].max())
        miny = float(meta["latitude"].min())
        maxy = float(meta["latitude"].max())
    return (minx - pad_deg, maxx + pad_deg, miny - pad_deg, maxy + pad_deg)


def _build_interpolation_grid(merged: pd.DataFrame, boundary_geom=None, nx: int = 220, ny: int = 180, pad_deg: float = 1.0):
    xmin, xmax, ymin, ymax = _get_plot_extent(merged, boundary_geom=boundary_geom, pad_deg=pad_deg)
    grid_x, grid_y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    return grid_x, grid_y


def _interpolate_quantile_surface(merged: pd.DataFrame, grid_x, grid_y, method: str = "thin_plate_spline"):
    x = merged["longitude"].to_numpy(dtype=float)
    y = merged["latitude"].to_numpy(dtype=float)
    z = merged["slope_per_decade"].to_numpy(dtype=float)
    method = (method or "thin_plate_spline").lower()

    if len(merged) < 4:
        return np.full_like(grid_x, np.nan, dtype=float)

    if method in {"thin_plate_spline", "rbf", "spline", "rbf_thin_plate"}:
        rbf = Rbf(x, y, z, function="thin_plate", smooth=0.0)
        return rbf(grid_x, grid_y)
    if method in {"multiquadric", "rbf_multiquadric"}:
        rbf = Rbf(x, y, z, function="multiquadric", smooth=0.0)
        return rbf(grid_x, grid_y)
    if method in {"linear", "cubic", "nearest"}:
        return griddata(np.column_stack([x, y]), z, (grid_x, grid_y), method=method)

    rbf = Rbf(x, y, z, function="thin_plate", smooth=0.0)
    return rbf(grid_x, grid_y)


def _mask_surface_to_boundary(grid_x, grid_y, surface, boundary_geom=None):
    if boundary_geom is None:
        return surface
    mask = contains_xy(boundary_geom, grid_x, grid_y)
    return np.where(mask, surface, np.nan)


def _draw_boundary(ax, boundary_geom):
    if boundary_geom is None:
        return
    geos = getattr(boundary_geom, "geoms", [boundary_geom])
    for geom in geos:
        try:
            x, y = geom.exterior.xy
            ax.plot(x, y, color="black", linewidth=0.8, zorder=3)
            for ring in geom.interiors:
                xi, yi = ring.xy
                ax.plot(xi, yi, color="black", linewidth=0.4, zorder=3)
        except Exception:
            continue


def _plot_spatial_trend_panel(ax, merged: pd.DataFrame, title: str, boundary_geom=None, interpolation_method: str = "thin_plate_spline"):
    if merged.empty:
        ax.set_axis_off()
        return None

    grid_x, grid_y = _build_interpolation_grid(merged, boundary_geom=boundary_geom)
    surface = _interpolate_quantile_surface(merged, grid_x, grid_y, method=interpolation_method)
    surface = _mask_surface_to_boundary(grid_x, grid_y, surface, boundary_geom=boundary_geom)

    finite = np.isfinite(surface)
    if finite.any():
        vmin = float(np.nanmin(surface))
        vmax = float(np.nanmax(surface))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        levels = np.linspace(vmin, vmax, 14)
        cf = ax.contourf(grid_x, grid_y, surface, levels=levels, cmap="coolwarm", extend="both", zorder=1)
    else:
        cf = None

    _draw_boundary(ax, boundary_geom)

    ax.scatter(merged["longitude"], merged["latitude"], s=22, facecolors="none", edgecolors="black", linewidths=0.5, zorder=4)
    sig = merged.loc[merged["significant_95"].fillna(False)]
    if not sig.empty:
        ax.scatter(sig["longitude"], sig["latitude"], s=20, c="grey", edgecolors="black", linewidths=0.2, zorder=5)

    xmin, xmax, ymin, ymax = _get_plot_extent(merged, boundary_geom=boundary_geom, pad_deg=0.4)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    return cf


def _apply_group_filter(df: pd.DataFrame, group_filter: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    out = df.copy()
    if group_filter:
        for col, value in group_filter.items():
            if col in out.columns:
                out = out.loc[out[col].astype(str) == str(value)].copy()
    return out


def _trend_subset(trends_df: pd.DataFrame, series: str, group_filter: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    out = _apply_group_filter(trends_df, group_filter)
    return out.loc[out["series"] == series].copy()


def _plot_timeseries_panel(ax, data_df: pd.DataFrame, trends_df: pd.DataFrame, series: str, group_filter=None) -> None:
    plot_df = _apply_group_filter(data_df, group_filter)
    plot_df = plot_df[["year", series]].dropna().sort_values("year")
    if plot_df.empty:
        ax.set_axis_off()
        return

    ax.plot(plot_df["year"], plot_df[series], marker="o", linewidth=1.3, markersize=2.5)
    subset = _trend_subset(trends_df, series, group_filter)
    subset = subset[
        ((subset["model_type"] == "quantile") & (subset["quantile"].round(2).isin([0.10, 0.50, 0.90])))
        | (subset["model_type"] == "ols_mean")
    ].copy()

    text_lines = []
    for key in [("ols_mean", np.nan), ("quantile", 0.10), ("quantile", 0.50), ("quantile", 0.90)]:
        if key[0] == "ols_mean":
            row = subset.loc[subset["model_type"] == "ols_mean"]
            if row.empty:
                continue
            row = row.iloc[0]
            yhat = row["intercept_centered"] + row["slope_per_year"] * (plot_df["year"] - row["year_mean"])
            ax.plot(plot_df["year"], yhat, linestyle=":", linewidth=2.0)
            text_lines.append(f"OLS={row['slope_per_decade']:.2f}")
        else:
            tau = key[1]
            row = subset.loc[(subset["model_type"] == "quantile") & (subset["quantile"].round(2) == round(tau, 2))]
            if row.empty:
                continue
            row = row.iloc[0]
            yhat = row["intercept_centered"] + row["slope_per_year"] * (plot_df["year"] - row["year_mean"])
            ax.plot(plot_df["year"], yhat, linewidth=1.8)
            text_lines.append(f"q{int(tau * 100):02d}={row['slope_per_decade']:.2f}")

    ax.set_title(SERIES_PANEL_LABELS.get(series, _series_label(series)), fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel("Days/year")
    ax.grid(True, alpha=0.25)
    if text_lines:
        ax.text(
            0.03,
            0.97,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7),
        )


def _plot_quantile_profile_panel(ax, trends_df: pd.DataFrame, series: str, group_filter=None) -> None:
    subset = _trend_subset(trends_df, series, group_filter)
    q = subset.loc[subset["model_type"] == "quantile"].sort_values("quantile").copy()
    ols = subset.loc[subset["model_type"] == "ols_mean"].copy()
    if q.empty or ols.empty:
        ax.set_axis_off()
        return

    ax.scatter(q["quantile"], q["slope_per_decade"], s=12)
    ax.fill_between(
        q["quantile"].to_numpy(dtype=float),
        q["ci_low_per_decade"].to_numpy(dtype=float),
        q["ci_high_per_decade"].to_numpy(dtype=float),
        alpha=0.20,
    )

    mean_row = ols.iloc[0]
    ax.axhline(mean_row["slope_per_decade"], linestyle="-", linewidth=1.8)
    ax.axhline(mean_row["ci_low_per_decade"], linestyle="--", linewidth=1.2)
    ax.axhline(mean_row["ci_high_per_decade"], linestyle="--", linewidth=1.2)
    ax.axhline(0, linewidth=1.0)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Trend (days/decade)")
    ax.set_title(SERIES_PANEL_LABELS.get(series, _series_label(series)), fontsize=11)
    ax.grid(True, alpha=0.25)


def _plot_map_panel(ax, merged: pd.DataFrame, title: str, focal_station_ids=None) -> None:
    if merged.empty:
        ax.set_axis_off()
        return

    focal_ids = set(str(x) for x in focal_station_ids) if focal_station_ids is not None else None
    base = merged.copy()
    base["station_id"] = base["station_id"].astype(str)

    if focal_ids is None:
        sc = ax.scatter(base["longitude"], base["latitude"], c=base["slope_per_decade"], s=95)
        sig = base.loc[base["significant_95"]]
        if not sig.empty:
            ax.scatter(sig["longitude"], sig["latitude"], s=30, marker="x")
        for _, row in base.iterrows():
            ax.text(row["longitude"] + 0.12, row["latitude"] + 0.12, row["station_name"], fontsize=7.5)
        ax._climate_colorbar_target = sc
    else:
        others = base.loc[~base["station_id"].isin(focal_ids)].copy()
        focus = base.loc[base["station_id"].isin(focal_ids)].copy()
        if not others.empty:
            ax.scatter(others["longitude"], others["latitude"], s=60, alpha=0.25)
        if not focus.empty:
            sc = ax.scatter(focus["longitude"], focus["latitude"], c=focus["slope_per_decade"], s=140)
            sig = focus.loc[focus["significant_95"]]
            if not sig.empty:
                ax.scatter(sig["longitude"], sig["latitude"], s=40, marker="x")
            for _, row in focus.iterrows():
                ax.text(row["longitude"] + 0.12, row["latitude"] + 0.12, row["station_name"], fontsize=8)
            ax._climate_colorbar_target = sc

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.25)


def plot_network_timeseries_with_trends(network_df: pd.DataFrame, trends_df: pd.DataFrame, series: str, output_path: Path) -> None:
    plot_df = network_df[["year", series]].dropna()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(plot_df["year"], plot_df[series], marker="o", linewidth=1.5, label="Observed")

    subset = trends_df[
        (trends_df["series"] == series)
        & (((trends_df["model_type"] == "quantile") & (trends_df["quantile"].round(2).isin([0.10, 0.50, 0.90])))
           | (trends_df["model_type"] == "ols_mean"))
    ].copy()
    for _, row in subset.iterrows():
        yhat = row["intercept_centered"] + row["slope_per_year"] * (plot_df["year"] - row["year_mean"])
        if row["model_type"] == "ols_mean":
            label = "OLS mean"
            linestyle = ":"
            linewidth = 2.2
        else:
            label = f"QR q={row['quantile']:.2f}"
            linestyle = "-"
            linewidth = 2.0
        ax.plot(plot_df["year"], yhat, linestyle=linestyle, linewidth=linewidth, label=label)

    ax.set_title(f"Network mean {_series_label(series)} with OLS and Quantile Trends")
    ax.set_xlabel("Year")
    ax.set_ylabel("Days per year")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_quantile_slope_profile(trends_df: pd.DataFrame, series: str, output_path: Path) -> None:
    q = trends_df[(trends_df["series"] == series) & (trends_df["model_type"] == "quantile")].copy()
    ols = trends_df[(trends_df["series"] == series) & (trends_df["model_type"] == "ols_mean")].copy()
    if q.empty or ols.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    q = q.sort_values("quantile")
    ax.scatter(q["quantile"], q["slope_per_decade"], s=18, label="Quantile slope")
    ax.fill_between(q["quantile"].to_numpy(), q["ci_low_per_decade"].to_numpy(), q["ci_high_per_decade"].to_numpy(), alpha=0.2, label="95% CI (QR)")
    mean_row = ols.iloc[0]
    ax.axhline(mean_row["slope_per_decade"], linestyle="-", linewidth=2, label="OLS mean slope")
    ax.axhline(mean_row["ci_low_per_decade"], linestyle="--", linewidth=1.5, label="OLS 95% CI")
    ax.axhline(mean_row["ci_high_per_decade"], linestyle="--", linewidth=1.5)
    ax.axhline(0, linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Trend (days per decade)")
    ax.set_title(f"Quantile slope profile: {_series_label(series)}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_station_quantile_map(station_trends: pd.DataFrame, station_metadata: pd.DataFrame, series: str, output_path: Path) -> None:
    df = station_trends.copy()
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype(str)
    df = df[
        (df["series"] == series)
        & (df["model_type"] == "quantile")
        & (df["quantile"].round(2) == 0.90)
    ].copy()
    if df.empty:
        return

    meta = station_metadata[["station_id", "station_name", "latitude", "longitude"]].copy()
    meta["station_id"] = meta["station_id"].astype(str)
    merged = df.merge(meta, on="station_id", how="left", suffixes=("", "_meta"))
    merged["station_name"] = merged["station_name_meta"].fillna(merged["station_name"])
    merged = merged.dropna(subset=["longitude", "latitude"])
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(merged["longitude"], merged["latitude"], c=merged["slope_per_decade"], s=90)
    sig = merged[merged["significant_95"]]
    if not sig.empty:
        ax.scatter(sig["longitude"], sig["latitude"], s=25, marker="x", label="95% significant")
    for _, row in merged.iterrows():
        ax.text(row["longitude"] + 0.15, row["latitude"] + 0.15, row["station_name"], fontsize=8)
    plt.colorbar(sc, ax=ax, label="Trend (days per decade)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Station q=0.90 trend pattern: {_series_label(series)}")
    ax.grid(True, alpha=0.3)
    if not sig.empty:
        ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_style_fig1(data_df: pd.DataFrame, trends_df: pd.DataFrame, output_path: Path, suptitle: str, group_filter=None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, series in zip(axes.flat, SERIES_ORDER):
        _plot_timeseries_panel(ax, data_df, trends_df, series, group_filter=group_filter)
    fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_style_fig2(trends_df: pd.DataFrame, output_path: Path, suptitle: str, group_filter=None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, series in zip(axes.flat, SERIES_ORDER):
        _plot_quantile_profile_panel(ax, trends_df, series, group_filter=group_filter)
    fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_style_fig3_network(
    station_trends: pd.DataFrame,
    station_metadata: pd.DataFrame,
    output_path: Path,
    suptitle="Figure 3 adaptation: q=0.90 station trends",
    boundary_path: Optional[Path] = None,
    interpolation_method: str = "thin_plate_spline",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
    color_target = None
    meta = station_metadata[["station_id", "station_name", "latitude", "longitude"]].copy()
    meta["station_id"] = meta["station_id"].astype(str)
    boundary_geom = _load_boundary_geometry(boundary_path)

    df_all = station_trends.copy()
    if "station_id" in df_all.columns:
        df_all["station_id"] = df_all["station_id"].astype(str)

    for ax, series in zip(axes.flat, SERIES_ORDER):
        df = df_all[(df_all["series"] == series) & (df_all["model_type"] == "quantile") & (df_all["quantile"].round(2) == 0.90)].copy()
        merged = df.merge(meta, on="station_id", how="left", suffixes=("", "_meta"))
        merged["station_name"] = merged["station_name_meta"].fillna(merged["station_name"])
        merged = merged.dropna(subset=["longitude", "latitude"])
        color_target = _plot_spatial_trend_panel(
            ax,
            merged,
            SERIES_PANEL_LABELS.get(series, _series_label(series)),
            boundary_geom=boundary_geom,
            interpolation_method=interpolation_method,
        ) or color_target

    if color_target is not None:
        fig.colorbar(color_target, ax=axes.ravel().tolist(), shrink=0.88, label="Trend (days/decade)")
    fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_style_fig3_station_focus(station_trends: pd.DataFrame, station_metadata: pd.DataFrame, focal_station_id: str, focal_station_name: str, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    color_target = None
    meta = station_metadata[["station_id", "station_name", "latitude", "longitude"]].copy()
    meta["station_id"] = meta["station_id"].astype(str)

    df_all = station_trends.copy()
    if "station_id" in df_all.columns:
        df_all["station_id"] = df_all["station_id"].astype(str)

    for ax, series in zip(axes.flat, SERIES_ORDER):
        df = df_all[
            (df_all["series"] == series)
            & (df_all["model_type"] == "quantile")
            & (df_all["quantile"].round(2) == 0.90)
        ].copy()
        merged = df.merge(meta, on="station_id", how="left", suffixes=("", "_meta"))
        merged["station_name"] = merged["station_name_meta"].fillna(merged["station_name"])
        merged = merged.dropna(subset=["longitude", "latitude"])
        _plot_map_panel(ax, merged, SERIES_PANEL_LABELS.get(series, _series_label(series)), focal_station_ids=[str(focal_station_id)])
        if hasattr(ax, "_climate_colorbar_target"):
            color_target = ax._climate_colorbar_target
    if color_target is not None:
        fig.colorbar(color_target, ax=axes.ravel().tolist(), shrink=0.88, label="Trend (days/decade)")
    fig.suptitle(f"Figure 3 adaptation for station: {focal_station_name} ({focal_station_id})", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_paper_style_fig3_cluster_focus(station_trends: pd.DataFrame, station_metadata: pd.DataFrame, station_clusters: pd.DataFrame, focal_cluster: int, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    color_target = None
    focus_ids = station_clusters.loc[station_clusters["cluster"] == int(focal_cluster), "station_id"].astype(str).tolist()
    meta = station_metadata[["station_id", "station_name", "latitude", "longitude"]].copy()
    meta["station_id"] = meta["station_id"].astype(str)

    df_all = station_trends.copy()
    if "station_id" in df_all.columns:
        df_all["station_id"] = df_all["station_id"].astype(str)

    for ax, series in zip(axes.flat, SERIES_ORDER):
        df = df_all[
            (df_all["series"] == series)
            & (df_all["model_type"] == "quantile")
            & (df_all["quantile"].round(2) == 0.90)
        ].copy()
        merged = df.merge(meta, on="station_id", how="left", suffixes=("", "_meta"))
        merged["station_name"] = merged["station_name_meta"].fillna(merged["station_name"])
        merged = merged.dropna(subset=["longitude", "latitude"])
        _plot_map_panel(ax, merged, SERIES_PANEL_LABELS.get(series, _series_label(series)), focal_station_ids=focus_ids)
        if hasattr(ax, "_climate_colorbar_target"):
            color_target = ax._climate_colorbar_target
    if color_target is not None:
        fig.colorbar(color_target, ax=axes.ravel().tolist(), shrink=0.88, label="Trend (days/decade)")
    fig.suptitle(f"Figure 3 adaptation for cluster {int(focal_cluster)}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cluster_membership_map(station_clusters: pd.DataFrame, output_path: Path) -> None:
    plot_df = station_clusters.copy()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sc = ax.scatter(plot_df["longitude"], plot_df["latitude"], c=plot_df["cluster"], s=120)
    for _, row in plot_df.iterrows():
        ax.text(row["longitude"] + 0.14, row["latitude"] + 0.14, row["station_name"], fontsize=8)
    plt.colorbar(sc, ax=ax, label="Cluster")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Station clusters from Ward hierarchical clustering")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
