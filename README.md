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


## Clearing outputs
```Get-ChildItem outputs/figures, outputs/models, outputs/tables -Force -Recurse |
> Where-Object { $_.Name -ne '.keep' } |
> Remove-Item -Force -Recurse
```
## References
- Fan, L.-J., 2014: Quantile trends in temperature extremes in China. *Theoretical and Applied Climatology*, 115, 1–12, https://doi.org/10.1007/s00704-013-1018-4.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*, 58(301), 236–244. https://doi.org/10.1080/01621459.1963.10500845
- Scikit-learn documentation on hierarchical clustering: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
- Scikit-learn documentation on silhouette analysis: https://scikit-learn.org/stable/modules/clustering.html#silhouette-analysis
- Statsmodels documentation on quantile regression: https://www.statsmodels.org/stable/regression.html#quantile-regression
- Matplotlib documentation on plotting: https://matplotlib.org/stable/contents.html

