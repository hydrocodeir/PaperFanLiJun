from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def create_method_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "Reference climatology",
                "detail": "Daily 10th and 90th percentile thresholds computed per station and calendar day from the 1961-1990 reference period.",
            },
            {
                "component": "Indices",
                "detail": "Warm days, cool days, warm nights, and cool nights counted annually from threshold exceedances.",
            },
            {
                "component": "Trend method",
                "detail": "Ordinary least squares for the mean trend and quantile regression for conditional quantiles from 0.01 to 0.99.",
            },
            {
                "component": "Clustering",
                "detail": "Stations grouped using Ward hierarchical clustering on elevation, mean annual temperature, and mean annual diurnal temperature range; number of clusters selected with the silhouette score.",
            },
            {
                "component": "Regional adaptation",
                "detail": "In addition to the equal-weight six-station network mean, cluster-level composite series are analyzed and plotted with paper-style Figure 1, Figure 2, and Figure 3 products.",
            },
            {
                "component": "Significance",
                "detail": "95% confidence intervals and p-values from fitted regression models; quantile significance is flagged when the regression confidence interval excludes zero.",
            },
        ]
    )


def compare_selected_quantiles(trends: pd.DataFrame, selected_quantiles: List[float]) -> pd.DataFrame:
    required_cols = {"model_type", "quantile"}
    if trends is None or trends.empty or not required_cols.issubset(trends.columns):
        return pd.DataFrame(columns=list(trends.columns) + ["quantile_label"] if isinstance(trends, pd.DataFrame) else ["quantile_label"])

    selected = trends[
        (trends["model_type"] == "quantile") & (trends["quantile"].round(2).isin([round(q, 2) for q in selected_quantiles]))
    ].copy()
    mean_df = trends[trends["model_type"] == "ols_mean"].copy()
    mean_df["quantile_label"] = "mean"

    selected["quantile_label"] = selected["quantile"].map(lambda q: f"q{q:.2f}")
    cols = [c for c in trends.columns if c not in {"quantile_label"}]
    out = pd.concat([mean_df[cols + ["quantile_label"]], selected[cols + ["quantile_label"]]], ignore_index=True)
    sort_cols = [c for c in ["series", "cluster", "station_id", "quantile_label"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def build_station_significance_summary(trends: pd.DataFrame, target_quantile: float = 0.90) -> pd.DataFrame:
    required_cols = {"model_type", "quantile"}
    if trends is None or trends.empty or not required_cols.issubset(trends.columns):
        return pd.DataFrame(columns=["series", "cluster", "station_id", "station_name", "cluster_id", "slope_per_decade", "significant_95"])

    q = trends[
        (trends["model_type"] == "quantile") &
        (trends["quantile"].round(2) == round(target_quantile, 2))
    ].copy()

    group_cols = [c for c in ["series", "cluster", "station_id", "station_name", "cluster_id"] if c in q.columns]
    if not group_cols:
        return q

    summary = q[group_cols + ["slope_per_decade", "significant_95"]].copy()
    return summary.sort_values(group_cols).reset_index(drop=True)


def build_quantile_spread_summary(
    trends: pd.DataFrame,
    quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90),
) -> pd.DataFrame:
    required_cols = {"series", "model_type", "quantile", "slope_per_decade", "significant_95"}
    if trends is None or trends.empty or not required_cols.issubset(trends.columns):
        return pd.DataFrame(
            columns=[
                "series",
                "slope_q10",
                "slope_q50",
                "slope_q90",
                "tail_amplification_q90_minus_q10",
                "tail_to_median_ratio_abs",
                "share_significant_selected_quantiles",
            ]
        )

    q_lo, q_mid, q_hi = [round(float(x), 2) for x in quantiles]
    selected = trends[
        (trends["model_type"] == "quantile")
        & (trends["quantile"].round(2).isin([q_lo, q_mid, q_hi]))
    ].copy()
    if selected.empty:
        return pd.DataFrame()

    selected["q_label"] = selected["quantile"].round(2).map(
        {q_lo: "slope_q10", q_mid: "slope_q50", q_hi: "slope_q90"}
    )
    pivot = (
        selected.pivot_table(index="series", columns="q_label", values="slope_per_decade", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    sig_share = (
        selected.groupby("series", as_index=False)["significant_95"]
        .mean()
        .rename(columns={"significant_95": "share_significant_selected_quantiles"})
    )
    out = pivot.merge(sig_share, on="series", how="left")

    for c in ["slope_q10", "slope_q50", "slope_q90"]:
        if c not in out.columns:
            out[c] = pd.NA
    out["tail_amplification_q90_minus_q10"] = out["slope_q90"] - out["slope_q10"]
    out["tail_to_median_ratio_abs"] = out["slope_q90"].abs() / out["slope_q50"].abs().clip(lower=1e-9)
    cols = [
        "series",
        "slope_q10",
        "slope_q50",
        "slope_q90",
        "tail_amplification_q90_minus_q10",
        "tail_to_median_ratio_abs",
        "share_significant_selected_quantiles",
    ]
    return out[cols].sort_values("series").reset_index(drop=True)


def build_station_extreme_trend_ranking(
    station_trends: pd.DataFrame,
    target_quantile: float = 0.90,
) -> pd.DataFrame:
    required_cols = {"station_id", "station_name", "series", "model_type", "quantile", "slope_per_decade", "significant_95"}
    if station_trends is None or station_trends.empty or not required_cols.issubset(station_trends.columns):
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "mean_abs_slope_per_decade",
                "share_significant_series",
                "direction_consistency",
                "rank_by_abs_slope",
            ]
        )

    q = station_trends[
        (station_trends["model_type"] == "quantile")
        & (station_trends["quantile"].round(2) == round(float(target_quantile), 2))
    ].copy()
    if q.empty:
        return pd.DataFrame()

    grouped = q.groupby(["station_id", "station_name"], as_index=False).agg(
        mean_abs_slope_per_decade=("slope_per_decade", lambda s: float(s.abs().mean())),
        share_significant_series=("significant_95", "mean"),
        n_positive=("slope_per_decade", lambda s: int((s > 0).sum())),
        n_negative=("slope_per_decade", lambda s: int((s < 0).sum())),
    )
    grouped["direction_consistency"] = (
        grouped[["n_positive", "n_negative"]].max(axis=1)
        / (grouped["n_positive"] + grouped["n_negative"]).clip(lower=1)
    )
    grouped = grouped.drop(columns=["n_positive", "n_negative"])
    grouped = grouped.sort_values(
        ["mean_abs_slope_per_decade", "share_significant_series"], ascending=[False, False]
    ).reset_index(drop=True)
    grouped["rank_by_abs_slope"] = grouped.index + 1
    return grouped


def build_station_discussion_table(
    station_trends: pd.DataFrame,
    station_break_summary: pd.DataFrame,
    target_quantile: float = 0.90,
) -> pd.DataFrame:
    ranking = build_station_extreme_trend_ranking(station_trends, target_quantile=target_quantile)
    if ranking.empty:
        return ranking

    q = station_trends[
        (station_trends["model_type"] == "quantile")
        & (station_trends["quantile"].round(2) == round(float(target_quantile), 2))
    ].copy()
    if q.empty:
        return ranking

    ci_bounds = q.groupby(["station_id", "station_name"], as_index=False).agg(
        mean_ci_high=("ci_high_per_decade", "mean"),
        mean_ci_low=("ci_low_per_decade", "mean"),
    )
    ci_bounds["mean_ci_width"] = ci_bounds["mean_ci_high"] - ci_bounds["mean_ci_low"]
    ci_summary = ci_bounds[["station_id", "station_name", "mean_ci_width"]].copy()
    warm = q.loc[q["series"] == "warm_days", ["station_id", "station_name", "slope_per_decade"]].rename(
        columns={"slope_per_decade": "slope_q90_warm_days"}
    )
    discussion = ranking.merge(ci_summary, on=["station_id", "station_name"], how="left")
    discussion = discussion.merge(warm, on=["station_id", "station_name"], how="left")
    if station_break_summary is not None and not station_break_summary.empty:
        discussion = discussion.merge(station_break_summary, on="station_id", how="left")
    if "n_breaks_total" not in discussion.columns:
        discussion["n_breaks_total"] = 0
    discussion["n_breaks_total"] = discussion["n_breaks_total"].fillna(0).astype(int)
    return discussion.sort_values("rank_by_abs_slope").reset_index(drop=True)


def build_journal_ready_results_table(
    network_trends: pd.DataFrame,
    cluster_trends: pd.DataFrame,
    selected_quantiles: List[float],
) -> pd.DataFrame:
    core_cols = [
        "series",
        "model_type",
        "quantile",
        "slope_per_decade",
        "ci_low_per_decade",
        "ci_high_per_decade",
        "p_value",
        "significant_95",
    ]
    if network_trends is None or network_trends.empty:
        return pd.DataFrame()
    if not set(core_cols).issubset(network_trends.columns):
        return pd.DataFrame()

    chosen_q = [round(float(q), 2) for q in selected_quantiles]
    keep_mask = (
        (network_trends["model_type"] == "ols_mean")
        | (
            (network_trends["model_type"] == "quantile")
            & (network_trends["quantile"].round(2).isin(chosen_q))
        )
    )
    network = network_trends.loc[keep_mask, core_cols + [c for c in ["network_id"] if c in network_trends.columns]].copy()
    network["analysis_level"] = "network"
    network["group_id"] = network.get("network_id", "network")
    network["quantile_label"] = np.where(
        network["model_type"] == "ols_mean",
        "mean",
        network["quantile"].map(lambda x: f"q{float(x):.2f}"),
    )

    cluster_table = pd.DataFrame(columns=network.columns.tolist() + ["cluster_range_slope", "cluster_sd_slope"])
    if cluster_trends is not None and not cluster_trends.empty and set(core_cols).issubset(cluster_trends.columns):
        c_keep = (
            (cluster_trends["model_type"] == "ols_mean")
            | (
                (cluster_trends["model_type"] == "quantile")
                & (cluster_trends["quantile"].round(2).isin(chosen_q))
            )
        )
        cluster = cluster_trends.loc[c_keep, core_cols + [c for c in ["cluster", "cluster_id"] if c in cluster_trends.columns]].copy()
        cluster["analysis_level"] = "cluster"
        cluster["group_id"] = cluster.get("cluster_id", cluster.get("cluster", "cluster"))
        cluster["quantile_label"] = np.where(
            cluster["model_type"] == "ols_mean",
            "mean",
            cluster["quantile"].map(lambda x: f"q{float(x):.2f}"),
        )
        comp = cluster.groupby(["series", "quantile_label"], as_index=False).agg(
            cluster_range_slope=("slope_per_decade", lambda s: float(s.max() - s.min())),
            cluster_sd_slope=("slope_per_decade", "std"),
        )
        cluster_table = cluster.merge(comp, on=["series", "quantile_label"], how="left")
    else:
        cluster_table = pd.DataFrame(columns=network.columns.tolist() + ["cluster_range_slope", "cluster_sd_slope"])

    network["cluster_range_slope"] = np.nan
    network["cluster_sd_slope"] = np.nan
    out = pd.concat([network, cluster_table], ignore_index=True, sort=False)
    out["effect_size"] = out["slope_per_decade"]
    out["ci_95"] = out["ci_low_per_decade"].map(lambda v: f"{v:.2f}") + " to " + out["ci_high_per_decade"].map(lambda v: f"{v:.2f}")
    out = out[
        [
            "analysis_level",
            "group_id",
            "series",
            "quantile_label",
            "effect_size",
            "p_value",
            "ci_95",
            "significant_95",
            "cluster_range_slope",
            "cluster_sd_slope",
        ]
    ].sort_values(["analysis_level", "series", "quantile_label", "group_id"]).reset_index(drop=True)
    return out


def build_journal_ready_wide_table(journal_ready_long: pd.DataFrame) -> pd.DataFrame:
    required = {"analysis_level", "group_id", "series", "quantile_label", "effect_size", "p_value", "ci_95", "significant_95"}
    if journal_ready_long is None or journal_ready_long.empty or not required.issubset(journal_ready_long.columns):
        return pd.DataFrame()

    df = journal_ready_long.copy()
    value_cols = ["effect_size", "p_value", "ci_95", "significant_95", "cluster_range_slope", "cluster_sd_slope"]
    value_cols = [c for c in value_cols if c in df.columns]
    wide = df.pivot_table(
        index=["analysis_level", "group_id", "series"],
        columns="quantile_label",
        values=value_cols,
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{q}" for metric, q in wide.columns]
    wide = wide.reset_index()

    preferred_q_order = ["mean", "q0.10", "q0.50", "q0.90"]
    leading = ["analysis_level", "group_id", "series"]
    ordered = leading + [
        c for q in preferred_q_order for c in wide.columns
        if c.endswith(f"__{q}")
    ] + [c for c in wide.columns if c not in leading and all(not c.endswith(f"__{q}") for q in preferred_q_order)]
    ordered = [c for c in ordered if c in wide.columns]
    return wide[ordered].sort_values(["analysis_level", "series", "group_id"]).reset_index(drop=True)
