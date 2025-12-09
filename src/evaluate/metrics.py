import numpy as np
import pandas as pd


def evaluate(y_true, y_pred, metrics=None, round_digits=2, **kwargs):
    """
    Evaluate model performance using a list of metric functions. Can
    pass functions directly, or a dict mapping a string metric to the
    function.
    """
    results = {}

    if metrics is None:
        return results

    for m in metrics:
        if isinstance(m, str):
            # Assuming 'metrics' is a registry for string lookups
            func = metrics.get(m)
            if func is None:
                raise ValueError(f"Metric '{m}' not found.")
            name = m
        else:
            func = m
            name = func.__name__

        # Calculate the metric value
        metric_value = func(y_true, y_pred, **kwargs)
        
        if round_digits is not None and round_digits >= 0:
            rounded_value = round(metric_value, round_digits)
        else:
            rounded_value = metric_value
        
        results[name] = float(rounded_value)

    return results


def pai(y_true, y_pred, region_grid, top_fraction=0.01, **kwargs):
    """
    Predictive Accuracy Index (PAI).

    PAI = (crimes in top hotspots / area of top hotspots)
          ------------------------------------------------
          (total crimes / total area)

    Args:
        y_true: Series or array of observed crimes in eval period
        y_pred: Series or array of predicted crime counts
        region_grid: GeoDataFrame containing the cells
        top_fraction: Fraction of grid cells to consider as hotspots

    Returns:
        float
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    n = len(y_pred)
    k = int(np.ceil(n * top_fraction))

    # top predicted cells
    top_idx = np.argsort(y_pred)[-k:]

    crimes_in_hot = y_true[top_idx].sum()
    total_crimes = y_true.sum()

    # area of cells
    areas = region_grid.geometry.area.values
    area_hot = areas[top_idx].sum()
    area_total = areas.sum()

    return (crimes_in_hot / area_hot) / (total_crimes / area_total)


def pei(y_true, y_pred, top_fraction=0.01, **kwargs):
    """
    Prediction Efficiency Index (PEI).

    PEI = (crimes captured by model's top hotspots) /
          (crimes captured by optimal (oracle) top hotspots)

    Args:
        y_true: Series or array of observed crimes in eval period.
        y_pred: Series or array of predicted crime counts.
        region_grid: Not used, but included for signature compatibility.
        top_fraction: Fraction of grid cells to consider as hotspots.

    Returns:
        float in [0, 1].
    """

    # ensure arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    n = len(y_pred)
    k = int(np.ceil(n * top_fraction))

    # model's chosen hotspots
    pred_top_idx = np.argsort(y_pred)[-k:]
    crimes_pred_top = y_true[pred_top_idx].sum()

    # oracle hotspots (based on true crimes)
    true_top_idx = np.argsort(y_true)[-k:]
    crimes_true_top = y_true[true_top_idx].sum()

    if crimes_true_top == 0:
        return 0.0

    return crimes_pred_top / crimes_true_top


def rri(y_true, y_pred, top_fraction=0.01, **kwargs):
    """
    Recapture Rate Index (RRI).

    RRI compares the crimes captured by the model's hotspots
    to the crimes expected under random selection of the same
    number of hotspots.

    RRI = (crimes in top predicted hotspots) /
          ((k / N) * total crimes)

    Args:
        y_true: Series or array of observed crimes in eval period.
        y_pred: Series or array of predicted crime counts.
        region_grid: Included for API compatibility, not used.
        top_fraction: Fraction of grid cells to consider as hotspots.

    Returns:
        float
    """

    # ensure arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    n = len(y_pred)
    k = int(np.ceil(n * top_fraction))

    # model-chosen hotspots
    top_idx = np.argsort(y_pred)[-k:]
    crimes_in_hot = y_true[top_idx].sum()

    total_crimes = y_true.sum()

    if total_crimes == 0:
        return 0.0

    # random expectation: (k/N) * total crimes
    expected_random = (k / n) * total_crimes

    # avoid divide-by-zero
    if expected_random == 0:
        return 0.0

    return crimes_in_hot / expected_random

