# Quantile Trends in Temperature Extremes for a 6-Station Synoptic Network

This project reproduces and adapts the methodology from **Fan Li-Jun (2014), "Quantile Trends in Temperature Extremes in China"** for a six-station synoptic temperature dataset and extends it with **Ward hierarchical clustering**.

## Expected inputs
Place these files at:

- `data/raw/data.csv`
- `data/raw/stationsInfo.csv`

### Required columns for `data.csv`
- `station_id`
- `station_name`
- `year`
- `month`
- `day`
- `tmin`
- `tmax`

Optional:
- `tmean`

### Required columns for `stationsInfo.csv`
- `station_id`
- `station_name`
- `latitude`
- `longitude`
- `elevation`

## What the pipeline does
1. Loads and quality-controls daily temperature observations.
2. Builds station-specific day-of-year 10th and 90th percentile thresholds using the 1961-1990 reference period.
3. Computes annual extreme temperature indices:
   - warm_days
   - cool_days
   - warm_nights
   - cool_nights
4. Aggregates a six-station network mean.
5. Builds clustering features from elevation, mean annual temperature, and mean annual diurnal temperature range.
6. Selects the optimal number of Ward clusters with the silhouette score.
7. Fits:
   - ordinary least squares mean trend
   - quantile regression trends for q = 0.01 to 0.99
8. Produces paper-style Figure 1, Figure 2, and Figure 3 outputs for:
   - the full network
   - each station
   - each cluster

## Run
```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Key outputs
- `data/processed/annual_extreme_indices_station.csv`
- `data/processed/annual_extreme_indices_network.csv`
- `data/processed/annual_extreme_indices_cluster.csv`
- `outputs/tables/station_clusters.csv`
- `outputs/tables/cluster_metrics.csv`
- `outputs/tables/cluster_trend_results.csv`
- `outputs/figures/ward_dendrogram.png`
- `outputs/figures/ward_silhouette.png`
- `outputs/figures/clusters/`
- `outputs/models/`

## Notes on adaptation
- The original paper used 549 stations and 2x2 degree gridding before computing a China-wide mean.
- For this six-station project, the network mean is an equal-weight station average.
- The cluster extension is an added regionalization layer for the six-station network and is not part of the original paper.
- Missing daily values are not blindly imputed into extreme-event counts. Annual counts are coverage-adjusted when annual completeness passes the configured threshold.
