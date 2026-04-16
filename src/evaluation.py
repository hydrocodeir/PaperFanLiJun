from __future__ import annotations

from typing import List

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
