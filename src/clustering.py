from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

INDEX_COLUMNS = ["warm_days", "cool_days", "warm_nights", "cool_nights"]
ANNUAL_CLIMATE_COLUMNS = ["tmean_annual", "tmax_annual", "tmin_annual"]


def load_station_metadata(config: Dict) -> pd.DataFrame:
    path = Path(config["data"]["station_info_csv"])
    if not path.exists():
        raise FileNotFoundError(
            f"Station metadata file not found: {path}. Place stationinfo CSV there before running the pipeline."
        )

    df = pd.read_csv(path)
    rename_map = {
        "lat": "latitude",
        "lon": "longitude",
        "elev": "elevation",
        "height": "elevation",
    }
    df = df.rename(columns=rename_map)

    required = ["station_id", "station_name", "latitude", "longitude", "elevation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Station metadata is missing required columns: {', '.join(missing)}")

    df = df[required].copy()
    df["station_id"] = df["station_id"].astype(str)
    df["station_name"] = df["station_name"].astype(str)
    for c in ["latitude", "longitude", "elevation"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("station_id").reset_index(drop=True)


def prepare_station_features(annual: pd.DataFrame, station_metadata: pd.DataFrame) -> pd.DataFrame:
    annual = annual.copy()
    annual["station_id"] = annual["station_id"].astype(str)
    annual["mean_dtr_annual"] = annual["tmax_annual"] - annual["tmin_annual"]

    features = (
        annual.groupby("station_id", as_index=False)
        .agg(
            mean_temp=("tmean_annual", "mean"),
            mean_dtr=("mean_dtr_annual", "mean"),
            years_available=("year", "nunique"),
        )
    )

    merged = station_metadata.merge(features, on="station_id", how="left", validate="one_to_one")
    merged["feature_missing"] = merged[["elevation", "mean_temp", "mean_dtr"]].isna().any(axis=1)
    if merged["feature_missing"].any():
        missing_ids = ", ".join(merged.loc[merged["feature_missing"], "station_id"].astype(str).tolist())
        raise ValueError(f"Cannot cluster stations because required features are missing for: {missing_ids}")

    return merged


def optimize_ward_clusters(
    features: pd.DataFrame,
    min_k: int = 2,
    max_k: int = 5,
) -> Tuple[int, float, pd.DataFrame, np.ndarray, StandardScaler]:
    X = features[["elevation", "mean_temp", "mean_dtr"]].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k = max(min(max_k, len(features) - 1), min_k)
    results = []
    best_k = None
    best_score = -np.inf

    for k in range(min_k, max_k + 1):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        inertia_like = 0.0
        for cluster_id in np.unique(labels):
            cluster_points = X_scaled[labels == cluster_id]
            center = cluster_points.mean(axis=0)
            inertia_like += float(((cluster_points - center) ** 2).sum())
        results.append({"k": int(k), "silhouette": float(score), "within_cluster_ss": inertia_like})
        if score > best_score:
            best_score = float(score)
            best_k = int(k)

    return best_k, best_score, pd.DataFrame(results), X_scaled, scaler


def fit_ward_clustering(
    features: pd.DataFrame,
    n_clusters: int,
) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    X = features[["elevation", "mean_temp", "mean_dtr"]].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X_scaled)

    out = features.copy()
    out["cluster"] = labels.astype(int)

    cluster_rank = (
        out.groupby("cluster")["mean_temp"]
        .mean()
        .sort_values()
        .reset_index()
        .reset_index()
        .rename(columns={"index": "cluster_order"})
    )
    cluster_rank["cluster_final"] = cluster_rank["cluster_order"].astype(int)
    rank_map = dict(zip(cluster_rank["cluster"], cluster_rank["cluster_final"]))
    out["cluster"] = out["cluster"].map(rank_map).astype(int)

    return out.sort_values(["cluster", "station_id"]).reset_index(drop=True), X_scaled, scaler


def aggregate_cluster_mean_indices(annual: pd.DataFrame, station_clusters: pd.DataFrame) -> pd.DataFrame:
    station_clusters = station_clusters[["station_id", "cluster"]].copy()
    station_clusters["station_id"] = station_clusters["station_id"].astype(str)

    annual = annual.copy()
    annual["station_id"] = annual["station_id"].astype(str)
    merged = annual.merge(station_clusters, on="station_id", how="inner", validate="many_to_one")

    value_cols = INDEX_COLUMNS + ANNUAL_CLIMATE_COLUMNS
    frames = []
    for (cluster_id, year), g in merged.groupby(["cluster", "year"], as_index=False):
        row = {
            "cluster_id": f"cluster_{int(cluster_id)}",
            "cluster": int(cluster_id),
            "year": int(year),
            "n_stations": int(g["station_id"].nunique()),
        }
        for col in value_cols:
            row[col] = float(g[col].mean()) if g[col].notna().any() else np.nan
            row[f"{col}_count"] = int(g[col].notna().sum())
        frames.append(row)

    return pd.DataFrame(frames).sort_values(["cluster", "year"]).reset_index(drop=True)


def summarize_clusters(station_clusters: pd.DataFrame) -> pd.DataFrame:
    summary = (
        station_clusters.groupby("cluster", as_index=False)
        .agg(
            n_stations=("station_id", "nunique"),
            mean_elevation=("elevation", "mean"),
            mean_temp=("mean_temp", "mean"),
            mean_dtr=("mean_dtr", "mean"),
        )
    )
    member_map = (
        station_clusters.groupby("cluster")["station_name"]
        .apply(lambda s: ", ".join(sorted(pd.Series(s).astype(str).tolist())))
        .reset_index(name="member_stations")
    )
    return summary.merge(member_map, on="cluster", how="left").sort_values("cluster").reset_index(drop=True)


def plot_ward_dendrogram(features: pd.DataFrame, output_path: Path) -> None:
    X = features[["elevation", "mean_temp", "mean_dtr"]].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    linkage_matrix = linkage(X_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{sid}\n{name}" for sid, name in zip(features["station_id"], features["station_name"])]
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=20, ax=ax)
    ax.set_title("Ward hierarchical clustering dendrogram")
    ax.set_xlabel("Station")
    ax.set_ylabel("Linkage distance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_silhouette_curve(metrics_df: pd.DataFrame, output_path: Path) -> None:
    if metrics_df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(metrics_df["k"], metrics_df["within_cluster_ss"], marker="o")
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Within-cluster SS")

    ax2 = ax1.twinx()
    ax2.plot(metrics_df["k"], metrics_df["silhouette"], marker="s", linestyle="--")
    ax2.set_ylabel("Silhouette score")

    ax1.set_title("Ward cluster optimization")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cluster_feature_space(station_clusters: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sc1 = ax.scatter(
        station_clusters["mean_temp"],
        station_clusters["mean_dtr"],
        c=station_clusters["cluster"],
        s=110,
    )
    for _, row in station_clusters.iterrows():
        ax.text(row["mean_temp"] + 0.03, row["mean_dtr"] + 0.03, str(row["station_name"]), fontsize=8)
    ax.set_xlabel("Mean annual temperature (C)")
    ax.set_ylabel("Mean annual DTR (C)")
    ax.set_title("Thermal feature space")

    ax = axes[1]
    sc2 = ax.scatter(
        station_clusters["elevation"],
        station_clusters["mean_temp"],
        c=station_clusters["cluster"],
        s=110,
    )
    for _, row in station_clusters.iterrows():
        ax.text(row["elevation"] + 5, row["mean_temp"] + 0.03, str(row["station_name"]), fontsize=8)
    ax.set_xlabel("Elevation (m)")
    ax.set_ylabel("Mean annual temperature (C)")
    ax.set_title("Elevation-temperature space")

    fig.colorbar(sc2, ax=axes.ravel().tolist(), shrink=0.85, label="Cluster")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
