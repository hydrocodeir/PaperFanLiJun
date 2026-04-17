from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import IterationLimitWarning


def _prep_series(year: pd.Series, values: pd.Series, min_obs: int = 10) -> pd.DataFrame:
    data = pd.DataFrame({"year": year, "value": values}).dropna().copy()
    if len(data) < int(min_obs):
        return data
    data["year_centered"] = data["year"] - data["year"].mean()
    return data


def _fit_ols(year: pd.Series, values: pd.Series, min_obs: int = 10):
    data = _prep_series(year, values, min_obs=min_obs)
    if len(data) < int(min_obs):
        return None, None
    X = sm.add_constant(data["year_centered"])
    model = sm.OLS(data["value"], X).fit()
    return model, data["year"].mean()


def _fit_quantile(year: pd.Series, values: pd.Series, tau: float, min_obs: int = 10):
    data = _prep_series(year, values, min_obs=min_obs)
    if len(data) < int(min_obs):
        return None, None
    X = sm.add_constant(data["year_centered"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IterationLimitWarning)
        model = sm.QuantReg(data["value"], X).fit(q=tau, max_iter=600)
    return model, data["year"].mean()


def _fit_one_group_series(task):
    group_key, group_dict, value_col, years, values, taus, min_obs = task
    series = pd.DataFrame({"year": years, value_col: values}).dropna()
    if len(series) < int(min_obs):
        return [], {}

    results = []
    model_entries = {}

    ols, year_mean = _fit_ols(series["year"], series[value_col], min_obs=min_obs)
    if ols is not None:
        ci = ols.conf_int().loc["year_centered"].tolist()
        results.append(
            {
                **group_dict,
                "series": value_col,
                "model_type": "ols_mean",
                "quantile": np.nan,
                "slope_per_year": float(ols.params["year_centered"]),
                "slope_per_decade": float(ols.params["year_centered"] * 10),
                "intercept_centered": float(ols.params["const"]),
                "year_mean": float(year_mean),
                "ci_low_per_year": float(ci[0]),
                "ci_high_per_year": float(ci[1]),
                "ci_low_per_decade": float(ci[0] * 10),
                "ci_high_per_decade": float(ci[1] * 10),
                "p_value": float(ols.pvalues["year_centered"]),
                "significant_95": bool(ols.pvalues["year_centered"] < 0.05),
                "r2_or_pr2": float(ols.rsquared),
                "n_obs": int(ols.nobs),
            }
        )
        model_entries[f"{value_col}__ols_mean"] = {"model": ols, "year_mean": float(year_mean)}

    for tau in taus:
        qr, year_mean = _fit_quantile(series["year"], series[value_col], tau=tau, min_obs=min_obs)
        if qr is None:
            continue
        ci = qr.conf_int().loc["year_centered"].tolist()
        pval = qr.pvalues.get("year_centered", np.nan)
        results.append(
            {
                **group_dict,
                "series": value_col,
                "model_type": "quantile",
                "quantile": float(tau),
                "slope_per_year": float(qr.params["year_centered"]),
                "slope_per_decade": float(qr.params["year_centered"] * 10),
                "intercept_centered": float(qr.params["const"]),
                "year_mean": float(year_mean),
                "ci_low_per_year": float(ci[0]),
                "ci_high_per_year": float(ci[1]),
                "ci_low_per_decade": float(ci[0] * 10),
                "ci_high_per_decade": float(ci[1] * 10),
                "p_value": float(pval) if pd.notna(pval) else np.nan,
                "significant_95": bool((ci[0] > 0) or (ci[1] < 0)),
                "r2_or_pr2": float(getattr(qr, "prsquared", np.nan)),
                "n_obs": int(qr.nobs),
            }
        )
        model_entries[f"{value_col}__quantile_{tau:.2f}"] = {"model": qr, "year_mean": float(year_mean)}

    return [{"group_key": group_key, **r} for r in results], {group_key: model_entries}


def fit_trend_suite(
    df: pd.DataFrame,
    value_columns: List[str],
    group_columns: List[str],
    taus: List[float],
    n_jobs: int = 1,
    min_obs: int = 10,
) -> Tuple[pd.DataFrame, Dict]:
    tasks = []
    for group_key, g in df.groupby(group_columns):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_dict = dict(zip(group_columns, group_key))
        for value_col in value_columns:
            series = g[["year", value_col]].dropna()
            if len(series) < int(min_obs):
                continue
            tasks.append((group_key, group_dict, value_col, series["year"].tolist(), series[value_col].tolist(), taus, int(min_obs)))

    all_results = []
    model_store: Dict = {}
    if n_jobs is None or int(n_jobs) <= 1 or len(tasks) <= 1:
        outputs = [_fit_one_group_series(task) for task in tasks]
    else:
        workers = min(int(n_jobs), len(tasks))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outputs = list(ex.map(_fit_one_group_series, tasks))

    for result_rows, model_chunk in outputs:
        all_results.extend(result_rows)
        for group_key, entries in model_chunk.items():
            model_store.setdefault(group_key, {}).update(entries)

    results_df = pd.DataFrame(all_results)
    if "group_key" in results_df.columns:
        results_df = results_df.drop(columns=["group_key"])
    return results_df, model_store


def save_model_store(model_store: Dict, output_path: Path) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(model_store, f)
