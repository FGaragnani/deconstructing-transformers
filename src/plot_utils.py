"""Plotting utilities for time series forecasting

Provides helpers to:
- reconstruct absolute predictions from predicted deltas
- flatten/concat prediction windows robustly (handles scalar outputs)
- plot full series with predicted windows
- plot a single forecast window (input + actual + predicted)

All functions work with numpy arrays and are independent of PyTorch.
"""
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def _ensure_1d(arr):
    a = np.asarray(arr)
    if a.ndim == 0:
        return a.reshape(-1)
    return a.reshape(-1)


def flatten_prediction_windows(predictions: List[np.ndarray]) -> np.ndarray:
    """Concatenate a list of prediction windows into a 1D array.

    Each element in `predictions` can be a scalar, 1D or 2D array. This
    function ensures every window is flattened and then concatenated.
    """
    if len(predictions) == 0:
        return np.array([])
    flat = [ _ensure_1d(p) for p in predictions ]
    return np.concatenate(flat)


def reconstruct_from_deltas(predictions: List[np.ndarray], series: np.ndarray, seq_len: int, pred_len: int) -> np.ndarray:
    """Reconstruct an absolute-valued prediction series from predicted deltas.

    - predictions: list of windows (each window: predicted deltas for that window)
    - series: original (possibly normalized) full series, used to get the starting value for each window
    - seq_len: length of the input context used for each window
    - pred_len: nominal prediction horizon used when sliding windows were generated

    Returns a numpy array same length as `series` where predicted timesteps are filled and other locations are NaN.
    This function is robust to windows of length 1.
    """
    prediction_series = np.full(len(series), np.nan)
    for idx, pred_window in enumerate(predictions):
        pw = _ensure_1d(pred_window)
        start = seq_len + idx * pred_len
        if start <= 0 or start - 1 >= len(series):
            continue
        last_val = float(series[start - 1])
        abs_pred = []
        for d in pw:
            last_val = last_val + float(d)
            abs_pred.append(last_val)
        end = start + len(abs_pred)
        # clip to series bounds
        if start < len(series):
            prediction_series[start: min(end, len(series))] = abs_pred[: max(0, len(series) - start)]
    return prediction_series


def plot_series_with_predictions(series: np.ndarray, prediction_series: np.ndarray, train_percentage: float = 0.8,
                                 ax: Optional[Axes] = None, title: str = "Original vs Predicted Series") -> Axes:
    """Plot the series and an overlaid prediction_series (both numpy arrays).

    - series: full original series (1D)
    - prediction_series: same length as series; contains predicted values or NaN where no prediction is available
    - train_percentage: used to highlight the test region
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(series))
    ax.plot(x, series, 'b-', label='Input Series')
    test_start = len(series) - int((1 - train_percentage) * len(series))
    ax.plot(np.arange(test_start, len(series)), series[test_start:], 'g-', label='Test Series', linewidth=2)
    ax.plot(x, prediction_series, 'r--', label='Predicted Series')
    ax.set_title(title)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_forecast_window(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, delta: bool = False,
                         ax: Optional[Axes] = None, show_legend: bool = True) -> Axes:
    """Plot a single forecast window: input context, true future and predicted future.

    - X: input context, shape (seq_len,) or (seq_len, 1)
    - y_true: ground-truth future values, shape (pred_len,) or (pred_len, 1)
    - y_pred: model output for the future. If delta=True, y_pred is interpreted as deltas and
      reconstructed starting from X[-1]. If not delta, it's plotted directly.

    The function returns the matplotlib Axes.
    """
    X = _ensure_1d(X)
    y_true = _ensure_1d(y_true)
    y_pred = _ensure_1d(y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    seq_len = len(X)

    # Plot the context
    ax.plot(range(seq_len), X, 'b-', label='Input', linewidth=2)

    # True future
    ax.plot(range(seq_len - 1, seq_len - 1 + len(y_true) + 1), [X[-1]] + list(y_true), 'g-', label='Actual', linewidth=2, marker='o')

    # Predicted future: reconstruct if delta
    if delta:
        last = float(X[-1])
        abs_pred = []
        for d in y_pred:
            last = last + float(d)
            abs_pred.append(last)
        ax.plot(range(seq_len - 1, seq_len - 1 + len(abs_pred) + 1), [X[-1]] + list(abs_pred), 'r--', label='Predicted', linewidth=2, marker='^')
    else:
        ax.plot(range(seq_len - 1, seq_len - 1 + len(y_pred) + 1), [X[-1]] + list(y_pred), 'r--', label='Predicted', linewidth=2, marker='^')

    if show_legend:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
