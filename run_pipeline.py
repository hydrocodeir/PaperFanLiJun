from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import json
from pathlib import Path
import re
import shutil
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import preprocess_temperature_data
from feature_engineering import (
    INDEX_SPECS,
    apply_thresholds_and_compute_indices,
    compute_daily_percentile_thresholds,
    compute_network_mean_indices,
)
from clustering import (
    aggregate_cluster_mean_indices,
    fit_ward_clustering,
    load_station_metadata,
    optimize_ward_clusters,
    plot_cluster_feature_space,
    plot_silhouette_curve,
    plot_ward_dendrogram,
    prepare_station_features,
    summarize_clusters,
)
from modeling import fit_trend_suite, save_model_store
from evaluation import (
    build_station_significance_summary,
    compare_selected_quantiles,
    create_method_summary,
)
from visualization import (
    plot_cluster_membership_map,
    plot_paper_style_fig345_network_suite,
    plot_network_timeseries_with_trends,
    plot_paper_style_fig1,
    plot_paper_style_fig2,
    plot_paper_style_fig3_cluster_focus,
    plot_paper_style_fig3_station_focus,
    plot_quantile_slope_profile,
    plot_station_quantile_map,
)


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs() -> None:
    for rel in [
        "data/processed",
        "outputs/tables",
        "outputs/figures",
        "outputs/models",
        "outputs/figures/stations",
        "outputs/figures/clusters",
    ]:
        (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "group"


def _copy_if_missing(src: Path, dst: Path) -> None:
    if not dst.exists() and src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    config = load_config()
    ensure_output_dirs()

    _copy_if_missing(Path("/mnt/data/data.csv"), PROJECT_ROOT / config["data"]["input_csv"])
    _copy_if_missing(Path("/mnt/data/stationsInfo.csv"), PROJECT_ROOT / config["data"]["station_info_csv"])

    prep = preprocess_temperature_data(config)
    prep.daily.to_csv(PROJECT_ROOT / "data/processed/clean_daily_data.csv", index=False)
    prep.quality_summary.to_csv(PROJECT_ROOT / "outputs/tables/data_quality_summary.csv", index=False)
    if prep.issue_log is not None and not prep.issue_log.empty:
        prep.issue_log.to_csv(PROJECT_ROOT / "outputs/tables/preprocessing_issue_log.csv", index=False)

    thresholds = compute_daily_percentile_thresholds(prep.daily, config)
    thresholds.to_csv(PROJECT_ROOT / "data/processed/daily_thresholds_1961_1990.csv", index=False)

    annual = apply_thresholds_and_compute_indices(prep.daily, thresholds, config)
    annual.to_csv(PROJECT_ROOT / "data/processed/annual_extreme_indices_station.csv", index=False)

    network = compute_network_mean_indices(annual, config)
    network["network_id"] = config["methodology"].get("network_id", "station_network")
    network.to_csv(PROJECT_ROOT / "data/processed/annual_extreme_indices_network.csv", index=False)

    station_metadata = load_station_metadata(config)
    station_metadata.to_csv(PROJECT_ROOT / "outputs/tables/station_metadata_used.csv", index=False)

    station_features = prepare_station_features(annual, station_metadata)
    station_features.to_csv(PROJECT_ROOT / "outputs/tables/station_features_for_clustering.csv", index=False)

    clustering_cfg = config["clustering"]
    best_k, best_score, cluster_metrics, _, _ = optimize_ward_clusters(
        station_features,
        min_k=int(clustering_cfg["min_k"]),
        max_k=int(clustering_cfg["max_k"]),
    )
    cluster_metrics.to_csv(PROJECT_ROOT / "outputs/tables/cluster_metrics.csv", index=False)

    station_clusters, _, _ = fit_ward_clustering(station_features, n_clusters=best_k)
    station_clusters.to_csv(PROJECT_ROOT / "outputs/tables/station_clusters.csv", index=False)
    summarize_clusters(station_clusters).to_csv(PROJECT_ROOT / "outputs/tables/cluster_summary.csv", index=False)

    plot_ward_dendrogram(station_features, PROJECT_ROOT / "outputs/figures/ward_dendrogram.png")
    plot_silhouette_curve(cluster_metrics, PROJECT_ROOT / "outputs/figures/ward_silhouette.png")
    plot_cluster_feature_space(station_clusters, PROJECT_ROOT / "outputs/figures/cluster_feature_space.png")
    plot_cluster_membership_map(station_clusters, PROJECT_ROOT / "outputs/figures/cluster_membership_map.png")

    cluster_annual = aggregate_cluster_mean_indices(annual, station_clusters)
    cluster_annual.to_csv(PROJECT_ROOT / "data/processed/annual_extreme_indices_cluster.csv", index=False)

    series_cols = list(INDEX_SPECS.keys())
    network_tau_grid = [round(float(x), 2) for x in config["modeling"]["quantiles_full_grid"]]
    station_tau_grid = [round(float(x), 2) for x in config["modeling"].get("station_quantiles_full_grid", config["modeling"]["quantiles_full_grid"])]
    cluster_tau_grid = [round(float(x), 2) for x in config["modeling"].get("cluster_quantiles_full_grid", config["modeling"]["quantiles_full_grid"])]

    n_jobs = int(config["modeling"].get("n_jobs", 1))

    network_trends, network_models = fit_trend_suite(network, value_columns=series_cols, group_columns=["network_id"], taus=network_tau_grid, n_jobs=n_jobs)

    station_catalog = annual[["station_id", "station_name"]].drop_duplicates().sort_values(["station_id", "station_name"])
    station_trends, station_models = fit_trend_suite(
        annual,
        value_columns=series_cols,
        group_columns=["station_id", "station_name"],
        taus=station_tau_grid,
        n_jobs=n_jobs,
    )

    cluster_trends, cluster_models = fit_trend_suite(
        cluster_annual,
        value_columns=series_cols,
        group_columns=["cluster", "cluster_id"],
        taus=cluster_tau_grid,
        n_jobs=n_jobs,
    )

    network_trends.to_csv(PROJECT_ROOT / "outputs/tables/network_trend_results.csv", index=False)
    station_trends.to_csv(PROJECT_ROOT / "outputs/tables/station_trend_results.csv", index=False)
    cluster_trends.to_csv(PROJECT_ROOT / "outputs/tables/cluster_trend_results.csv", index=False)
    save_model_store(network_models, PROJECT_ROOT / "outputs/models/network_trend_models.pkl")
    save_model_store(station_models, PROJECT_ROOT / "outputs/models/station_trend_models.pkl")
    save_model_store(cluster_models, PROJECT_ROOT / "outputs/models/cluster_trend_models.pkl")

    selected_quantiles = [float(q) for q in config["modeling"]["selected_quantiles"]]
    compare_selected_quantiles(network_trends, selected_quantiles).to_csv(PROJECT_ROOT / "outputs/tables/network_selected_quantile_comparison.csv", index=False)
    compare_selected_quantiles(station_trends, selected_quantiles).to_csv(PROJECT_ROOT / "outputs/tables/station_selected_quantile_comparison.csv", index=False)
    compare_selected_quantiles(cluster_trends, selected_quantiles).to_csv(PROJECT_ROOT / "outputs/tables/cluster_selected_quantile_comparison.csv", index=False)

    build_station_significance_summary(station_trends, target_quantile=0.90).to_csv(PROJECT_ROOT / "outputs/tables/station_q90_significance_summary.csv", index=False)
    build_station_significance_summary(cluster_trends, target_quantile=0.90).to_csv(PROJECT_ROOT / "outputs/tables/cluster_q90_significance_summary.csv", index=False)
    create_method_summary().to_csv(PROJECT_ROOT / "outputs/tables/methodology_summary.csv", index=False)

    for series in series_cols:
        plot_network_timeseries_with_trends(network, network_trends, series, PROJECT_ROOT / f"outputs/figures/network_timeseries_{series}.png")
        plot_quantile_slope_profile(network_trends, series, PROJECT_ROOT / f"outputs/figures/network_quantile_profile_{series}.png")
        plot_station_quantile_map(station_trends, station_metadata, series, PROJECT_ROOT / f"outputs/figures/station_q90_map_{series}.png")

    plot_paper_style_fig1(
        network,
        network_trends,
        PROJECT_ROOT / "outputs/figures/paper_fig1_network.png",
        suptitle="Figure 1 adaptation: network mean indices",
        group_filter={"network_id": config["methodology"].get("network_id", "station_network")},
    )
    plot_paper_style_fig2(
        network_trends,
        PROJECT_ROOT / "outputs/figures/paper_fig2_network.png",
        suptitle="Figure 2 adaptation: network quantile slope profiles",
        group_filter={"network_id": config["methodology"].get("network_id", "station_network")},
    )
    plot_paper_style_fig345_network_suite(
        station_trends=station_trends,
        station_metadata=station_metadata,
        output_dir=PROJECT_ROOT / "outputs/figures",
        boundary_path=PROJECT_ROOT / config["spatial_visualization"]["iran_boundary_geojson"],
        interpolation_method="thin_plate_spline",
        interpolation_smooth=float(config["spatial_visualization"].get("interpolation_smooth", 0.35)),
        quantiles=((3, 0.90), (4, 0.50), (5, 0.10)),
    )

    for row in station_catalog.itertuples(index=False):
        station_id = str(row.station_id)
        station_name = str(row.station_name)
        station_slug = f"{station_id}_{_slugify(station_name)}"
        station_dir = PROJECT_ROOT / "outputs/figures/stations" / station_slug
        station_dir.mkdir(parents=True, exist_ok=True)
        group_filter = {"station_id": station_id, "station_name": station_name}
        station_annual = annual.loc[(annual["station_id"].astype(str) == station_id) & (annual["station_name"] == station_name)].copy()

        plot_paper_style_fig1(station_annual, station_trends, station_dir / "paper_fig1_station.png", suptitle=f"Figure 1 adaptation: {station_name} ({station_id})", group_filter=group_filter)
        plot_paper_style_fig2(station_trends, station_dir / "paper_fig2_station.png", suptitle=f"Figure 2 adaptation: {station_name} ({station_id})", group_filter=group_filter)
        plot_paper_style_fig3_station_focus(station_trends, station_metadata, focal_station_id=station_id, focal_station_name=station_name, output_path=station_dir / "paper_fig3_station.png")

    cluster_catalog = station_clusters[["cluster"]].drop_duplicates().sort_values("cluster")
    for row in cluster_catalog.itertuples(index=False):
        cluster_num = int(row.cluster)
        cluster_id = f"cluster_{cluster_num}"
        cluster_dir = PROJECT_ROOT / "outputs/figures/clusters" / cluster_id
        cluster_dir.mkdir(parents=True, exist_ok=True)
        group_filter = {"cluster": cluster_num, "cluster_id": cluster_id}
        cluster_df = cluster_annual.loc[cluster_annual["cluster"] == cluster_num].copy()

        plot_paper_style_fig1(cluster_df, cluster_trends, cluster_dir / "paper_fig1_cluster.png", suptitle=f"Figure 1 adaptation: {cluster_id}", group_filter=group_filter)
        plot_paper_style_fig2(cluster_trends, cluster_dir / "paper_fig2_cluster.png", suptitle=f"Figure 2 adaptation: {cluster_id}", group_filter=group_filter)
        plot_paper_style_fig3_cluster_focus(station_trends, station_metadata, station_clusters, focal_cluster=cluster_num, output_path=cluster_dir / "paper_fig3_cluster.png")

    summary = {
        "input_csv": config["data"]["input_csv"],
        "station_info_csv": config["data"]["station_info_csv"],
        "n_clean_daily_rows": int(len(prep.daily)),
        "n_station_year_rows": int(len(annual)),
        "n_network_year_rows": int(len(network)),
        "n_cluster_year_rows": int(len(cluster_annual)),
        "series_modeled": series_cols,
        "selected_quantiles": selected_quantiles,
        "station_quantiles_full_grid": station_tau_grid,
        "cluster_quantiles_full_grid": cluster_tau_grid,
        "generated_station_figure_sets": int(len(station_catalog)),
        "generated_cluster_figure_sets": int(len(cluster_catalog)),
        "clustering_method": "hierarchical_ward",
        "optimal_k": int(best_k),
        "best_silhouette": float(best_score),
    }
    with open(PROJECT_ROOT / "outputs/models/run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Pipeline finished successfully.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
