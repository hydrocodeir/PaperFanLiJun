from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import Rbf, griddata
from shapely import contains_xy
from shapely.geometry import Point
from shapely.ops import polygonize, unary_union as shp_unary_union

from plot_theme import PUB_DPI, apply_publication_defaults, save_figure

SERIES_ORDER = ["warm_days", "cool_days", "warm_nights", "cool_nights"]
SERIES_PANEL_LABELS = {
    "warm_days": "(a) Warm days",
    "cool_days": "(b) Cool days",
    "warm_nights": "(c) Warm nights",
    "cool_nights": "(d) Cool nights",
}

FIG3_LEVELS = np.arange(-14, 16, 2, dtype=float)
FIG3_CMAP = LinearSegmentedColormap.from_list(
    "fanli_fig3",
    [
        "#10207a",  # deep blue
        "#1f4aa8",
        "#3b7d2a",  # green
        "#8cbf26",
        "#f2f0d0",  # near-zero pale cream
        "#f0dd3d",  # yellow
        "#ef9b1a",
        "#d6281f",  # red
    ],
    N=256,
)
FIG3_NORM = BoundaryNorm(FIG3_LEVELS, FIG3_CMAP.N, clip=False)

apply_publication_defaults()


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
    gdf = gdf.to_crs(4326)
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union

    # Some boundary files are stored as line work (LineString/MultiLineString)
    # instead of Polygon geometries. Convert them to polygons for clipping.
    if geom.geom_type in {"LineString", "MultiLineString"}:
        polys = list(polygonize(geom))
        if polys:
            geom = shp_unary_union(polys)
    return geom


def _get_plot_extent(meta: pd.DataFrame, boundary_geom=None, pad_deg: float = 0.4):
    if boundary_geom is not None:
        minx, miny, maxx, maxy = boundary_geom.bounds
    else:
        minx = float(meta["longitude"].min())
        maxx = float(meta["longitude"].max())
        miny = float(meta["latitude"].min())
        maxy = float(meta["latitude"].max())
    return (minx - pad_deg, maxx + pad_deg, miny - pad_deg, maxy + pad_deg)


def _build_interpolation_grid(merged: pd.DataFrame, boundary_geom=None, nx: int = 320, ny: int = 320, pad_deg: float = 0.0):
    xmin, xmax, ymin, ymax = _get_plot_extent(merged, boundary_geom=boundary_geom, pad_deg=pad_deg)
    grid_x, grid_y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    return grid_x, grid_y


def _interpolate_quantile_surface(merged: pd.DataFrame, grid_x, grid_y, method: str = "thin_plate_spline", smooth: float = 0.35):
    x = merged["longitude"].to_numpy(dtype=float)
    y = merged["latitude"].to_numpy(dtype=float)
    z = merged["slope_per_decade"].to_numpy(dtype=float)
    method = (method or "thin_plate_spline").lower()

    if len(merged) < 4:
        return np.full_like(grid_x, np.nan, dtype=float)

    rbf_aliases = {
        "thin_plate_spline": "thin_plate",
        "rbf": "thin_plate",
        "spline": "thin_plate",
        "rbf_thin_plate": "thin_plate",
        "multiquadric": "multiquadric",
        "rbf_multiquadric": "multiquadric",
        "inverse": "inverse",
        "rbf_inverse": "inverse",
        "gaussian": "gaussian",
        "rbf_gaussian": "gaussian",
        "linear_rbf": "linear",
        "rbf_linear": "linear",
        "quintic": "quintic",
        "rbf_quintic": "quintic",
    }
    if method in rbf_aliases:
        rbf = Rbf(x, y, z, function=rbf_aliases[method], smooth=smooth)
        return rbf(grid_x, grid_y)
    if method in {"linear", "cubic", "nearest"}:
        surface = griddata(np.column_stack([x, y]), z, (grid_x, grid_y), method=method)
        if method != "nearest" and np.isnan(surface).any():
            nearest = griddata(np.column_stack([x, y]), z, (grid_x, grid_y), method="nearest")
            surface = np.where(np.isnan(surface), nearest, surface)
        return surface

    rbf = Rbf(x, y, z, function="thin_plate", smooth=smooth)
    return rbf(grid_x, grid_y)


def _mask_surface_to_boundary(grid_x, grid_y, surface, boundary_geom=None):
    if boundary_geom is None:
        return surface
    try:
        mask = contains_xy(boundary_geom, grid_x.ravel(), grid_y.ravel()).reshape(grid_x.shape)
    except Exception:
        mask = np.zeros_like(grid_x, dtype=bool)
    if not np.any(mask):
        # Fallback for geometry engines that fail on vectorized contains_xy.
        pts = [Point(float(x), float(y)) for x, y in zip(grid_x.ravel(), grid_y.ravel())]
        mask = np.array([boundary_geom.covers(pt) for pt in pts], dtype=bool).reshape(grid_x.shape)
    return np.where(mask, surface, np.nan)


def _draw_boundary(ax, boundary_geom, linewidth: float = 1.0):
    if boundary_geom is None:
        return
    geos = getattr(boundary_geom, "geoms", [boundary_geom])
    for geom in geos:
        try:
            x, y = geom.exterior.xy
            ax.plot(x, y, color="black", linewidth=linewidth, zorder=4)
            for ring in geom.interiors:
                xi, yi = ring.xy
                ax.plot(xi, yi, color="black", linewidth=max(0.35, linewidth * 0.4), zorder=4)
        except Exception:
            continue


def _format_geo_axis(ax, show_x: bool, show_y: bool) -> None:
    xticks = np.arange(45, 64, 2.5)
    yticks = np.arange(26, 41, 2.0)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    if show_x:
        ax.set_xticklabels([f"{x:.1f}°E" for x in xticks], fontsize=8)
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_yticklabels([f"{y:.0f}°N" for y in yticks], fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.tick_params(direction="out", length=3.5, width=0.8)


def _plot_spatial_trend_panel(
    ax,
    merged: pd.DataFrame,
    title: str,
    boundary_geom=None,
    interpolation_method: str = "thin_plate_spline",
    interpolation_smooth: float = 0.35,
    levels: np.ndarray = FIG3_LEVELS,
):
    if merged.empty:
        ax.set_axis_off()
        return None

    grid_x, grid_y = _build_interpolation_grid(merged, boundary_geom=boundary_geom)
    surface = _interpolate_quantile_surface(
        merged,
        grid_x,
        grid_y,
        method=interpolation_method,
        smooth=interpolation_smooth,
    )
    surface = _mask_surface_to_boundary(grid_x, grid_y, surface, boundary_geom=boundary_geom)

    cf = ax.contourf(
        grid_x,
        grid_y,
        surface,
        levels=levels,
        cmap=FIG3_CMAP,
        norm=FIG3_NORM,
        extend="both",
        antialiased=True,
        zorder=1,
    )

    _draw_boundary(ax, boundary_geom, linewidth=1.0)

    sig = merged.loc[merged["significant_95"].fillna(False)].copy()
    if not sig.empty:
        ax.scatter(
            sig["longitude"],
            sig["latitude"],
            s=10,
            c="#bdbdbd",
            edgecolors="none",
            zorder=5,
        )

    xmin, xmax, ymin, ymax = _get_plot_extent(merged, boundary_geom=boundary_geom, pad_deg=0.1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
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
    if trends_df is None or trends_df.empty or "series" not in trends_df.columns:
        return pd.DataFrame()
    out = _apply_group_filter(trends_df, group_filter)
    return out.loc[out["series"] == series].copy()


def _prepare_station_series_map_data(
    station_trends: pd.DataFrame,
    station_metadata: pd.DataFrame,
    series: str,
    target_quantile: float,
) -> pd.DataFrame:
    required = {"series", "model_type", "quantile", "station_id"}
    if station_trends is None or station_trends.empty or not required.issubset(station_trends.columns):
        return pd.DataFrame()

    df = station_trends.copy()
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype(str)

    q = round(float(target_quantile), 2)
    df = df[
        (df["series"] == series)
        & (df["model_type"] == "quantile")
        & (df["quantile"].round(2) == q)
    ].copy()
    if df.empty:
        return df

    meta = station_metadata[["station_id", "station_name", "latitude", "longitude"]].copy()
    meta["station_id"] = meta["station_id"].astype(str)
    merged = df.merge(meta, on="station_id", how="left", suffixes=("", "_meta"))
    merged["station_name"] = merged["station_name_meta"].fillna(merged["station_name"])
    return merged.dropna(subset=["longitude", "latitude"]).copy()


def _plot_timeseries_panel(ax, data_df: pd.DataFrame, trends_df: pd.DataFrame, series: str, group_filter=None) -> None:
    plot_df = _apply_group_filter(data_df, group_filter)
    plot_df = plot_df[["year", series]].dropna().sort_values("year")
    if plot_df.empty:
        ax.set_axis_off()
        return

    ax.plot(plot_df["year"], plot_df[series], marker="o", linewidth=1.3, markersize=2.5)
    subset = _trend_subset(trends_df, series, group_filter)
    required = {"model_type", "quantile"}
    if subset.empty or not required.issubset(subset.columns):
        ax.set_title(SERIES_PANEL_LABELS.get(series, _series_label(series)), fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel("Days/year")
        ax.grid(True, alpha=0.25)
        return
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
    required = {"model_type", "quantile", "slope_per_decade", "ci_low_per_decade", "ci_high_per_decade"}
    if subset.empty or not required.issubset(subset.columns):
        ax.set_axis_off()
        return
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
            sc = ax.scatter(focus["longitude"], focus["latitude"], c=focus["slope_per_decade"], s=140, edgecolors="black", linewidths=0.5)
            sig = focus.loc[focus["significant_95"]]
            if not sig.empty:
                ax.scatter(sig["longitude"], sig["latitude"], s=45, marker="x")
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

    subset = pd.DataFrame()
    required = {"series", "model_type", "quantile"}
    if trends_df is not None and not trends_df.empty and required.issubset(trends_df.columns):
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
    save_figure(fig, output_path)


def plot_quantile_slope_profile(trends_df: pd.DataFrame, series: str, output_path: Path) -> None:
    required = {"series", "model_type", "quantile"}
    if trends_df is None or trends_df.empty or not required.issubset(trends_df.columns):
        return

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
    save_figure(fig, output_path)


def plot_station_quantile_map(station_trends: pd.DataFrame, station_metadata: pd.DataFrame, series: str, output_path: Path) -> None:
    merged = _prepare_station_series_map_data(
        station_trends=station_trends,
        station_metadata=station_metadata,
        series=series,
        target_quantile=0.90,
    )
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
    save_figure(fig, output_path)


def plot_paper_style_fig1(data_df: pd.DataFrame, trends_df: pd.DataFrame, output_path: Path, suptitle: str, group_filter=None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, series in zip(axes.flat, SERIES_ORDER):
        _plot_timeseries_panel(ax, data_df, trends_df, series, group_filter=group_filter)
    fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_path)


def plot_paper_style_fig2(trends_df: pd.DataFrame, output_path: Path, suptitle: str, group_filter=None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, series in zip(axes.flat, SERIES_ORDER):
        _plot_quantile_profile_panel(ax, trends_df, series, group_filter=group_filter)
    fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_path)


def plot_paper_style_fig3_network(
    station_trends: pd.DataFrame,
    station_metadata: pd.DataFrame,
    output_path: Path,
    suptitle="Figure 3 adaptation: q=0.90 station trends",
    boundary_path: Optional[Path] = None,
    interpolation_method: str = "thin_plate_spline",
    interpolation_smooth: float = 0.35,
    target_quantile: float = 0.90,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6), constrained_layout=False)
    fig.patch.set_facecolor("white")
    if boundary_path is None:
        default_boundary = Path("data/raw/Iran.geojson")
        boundary_path = default_boundary if default_boundary.exists() else None
    boundary_geom = _load_boundary_geometry(boundary_path)

    last_cf = None
    target_q = round(float(target_quantile), 2)
    for ax, series in zip(axes.flat, SERIES_ORDER):
        merged = _prepare_station_series_map_data(
            station_trends=station_trends,
            station_metadata=station_metadata,
            series=series,
            target_quantile=target_q,
        )
        last_cf = _plot_spatial_trend_panel(
            ax,
            merged,
            SERIES_PANEL_LABELS.get(series, _series_label(series)),
            boundary_geom=boundary_geom,
            interpolation_method=interpolation_method,
            interpolation_smooth=interpolation_smooth,
            levels=FIG3_LEVELS,
        )
        row_idx, col_idx = divmod(SERIES_ORDER.index(series), 2)
        _format_geo_axis(ax, show_x=(row_idx == 1), show_y=(col_idx == 0))
        ax.set_facecolor("#f4f4f4")
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

    fig.suptitle(suptitle, fontsize=14, y=0.975, fontweight="semibold")
    cax = fig.add_axes([0.14, 0.055, 0.72, 0.028])
    cbar = fig.colorbar(last_cf, cax=cax, orientation="horizontal", ticks=FIG3_LEVELS)
    cbar.set_label("Trend (days per decade)", fontsize=12, fontweight="medium")
    cbar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.12, wspace=0.06, hspace=0.12)
    save_figure(fig, output_path, dpi=PUB_DPI)


def plot_paper_style_fig345_network_suite(
    station_trends: pd.DataFrame,
    station_metadata: pd.DataFrame,
    output_dir: Path,
    boundary_path: Optional[Path] = None,
    interpolation_method: str = "thin_plate_spline",
    interpolation_smooth: float = 0.35,
    quantiles: Sequence[Tuple[int, float]] = ((3, 0.90), (4, 0.50), (5, 0.10)),
) -> None:
    output_dir = Path(output_dir)
    for fig_num, q in quantiles:
        plot_paper_style_fig3_network(
            station_trends=station_trends,
            station_metadata=station_metadata,
            output_path=output_dir / f"paper_fig{int(fig_num)}_network.png",
            suptitle=f"Figure {int(fig_num)} adaptation: q={float(q):.2f} station trends for the network",
            boundary_path=boundary_path,
            interpolation_method=interpolation_method,
            interpolation_smooth=interpolation_smooth,
            target_quantile=float(q),
        )


def plot_paper_style_fig3_station_focus(station_trends: pd.DataFrame, station_metadata: pd.DataFrame, focal_station_id: str, focal_station_name: str, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    color_target = None

    for ax, series in zip(axes.flat, SERIES_ORDER):
        merged = _prepare_station_series_map_data(
            station_trends=station_trends,
            station_metadata=station_metadata,
            series=series,
            target_quantile=0.90,
        )
        _plot_map_panel(ax, merged, SERIES_PANEL_LABELS.get(series, _series_label(series)), focal_station_ids=[str(focal_station_id)])
        if hasattr(ax, "_climate_colorbar_target"):
            color_target = ax._climate_colorbar_target
    if color_target is not None:
        fig.colorbar(color_target, ax=axes.ravel().tolist(), shrink=0.88, label="Trend (days/decade)")
    fig.suptitle(f"Figure 3 adaptation for station: {focal_station_name} ({focal_station_id})", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_path)


def plot_paper_style_fig3_cluster_focus(station_trends: pd.DataFrame, station_metadata: pd.DataFrame, station_clusters: pd.DataFrame, focal_cluster: int, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    color_target = None
    focus_ids = station_clusters.loc[station_clusters["cluster"] == int(focal_cluster), "station_id"].astype(str).tolist()

    for ax, series in zip(axes.flat, SERIES_ORDER):
        merged = _prepare_station_series_map_data(
            station_trends=station_trends,
            station_metadata=station_metadata,
            series=series,
            target_quantile=0.90,
        )
        _plot_map_panel(ax, merged, SERIES_PANEL_LABELS.get(series, _series_label(series)), focal_station_ids=focus_ids)
        if hasattr(ax, "_climate_colorbar_target"):
            color_target = ax._climate_colorbar_target
    if color_target is not None:
        fig.colorbar(color_target, ax=axes.ravel().tolist(), shrink=0.88, label="Trend (days/decade)")
    fig.suptitle(f"Figure 3 adaptation for cluster {int(focal_cluster)}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_path)


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
    save_figure(fig, output_path)
