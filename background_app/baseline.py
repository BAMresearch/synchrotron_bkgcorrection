from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .config import BaselineSettings


class BaselineError(RuntimeError):
    """Raised when a baseline cannot be calculated reliably."""


@lru_cache(maxsize=32)
def penalty_matrix(length: int) -> sparse.csc_matrix:
    if length < 3:
        raise BaselineError("At least three points are required for baseline correction")
    difference = sparse.eye(length, format="csc")
    difference = difference[1:] - difference[:-1]
    difference = difference[1:] - difference[:-1]
    return (difference.T @ difference).tocsc()


def fit_baseline(
    y: np.ndarray,
    settings: BaselineSettings,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    values = np.asarray(y, dtype=np.float64)
    _validate(values, settings)
    if settings.algorithm == "als":
        result = asymmetric_least_squares(values, settings, cancelled)
    elif settings.algorithm == "arpls":
        result = arpls(values, settings, cancelled)
    else:
        raise BaselineError(f"Unknown baseline algorithm: {settings.algorithm}")
    if not np.isfinite(result).all():
        raise BaselineError("Baseline calculation produced non-finite values")
    return result


def asymmetric_least_squares(
    y: np.ndarray,
    settings: BaselineSettings,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    p = settings.asymmetry
    weights = np.ones(y.size, dtype=np.float64)
    penalty = settings.lam * penalty_matrix(y.size)
    baseline = y.copy()

    for _ in range(settings.max_iterations):
        if cancelled and cancelled():
            raise InterruptedError("Processing cancelled")
        system = sparse.diags(weights, format="csc") + penalty
        baseline_new = _safe_solve(system, weights * y)
        new_weights = np.where(y > baseline_new, p, 1.0 - p)
        relative_change = np.linalg.norm(baseline_new - baseline) / max(np.linalg.norm(baseline), np.finfo(float).eps)
        baseline = baseline_new
        if np.array_equal(new_weights, weights) or relative_change <= settings.tolerance:
            break
        weights = new_weights
    return baseline


def arpls(
    y: np.ndarray,
    settings: BaselineSettings,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    weights = np.ones(y.size, dtype=np.float64)
    penalty = settings.lam * penalty_matrix(y.size)
    baseline = y.copy()

    for _ in range(settings.max_iterations):
        if cancelled and cancelled():
            raise InterruptedError("Processing cancelled")
        system = sparse.diags(weights, format="csc") + penalty
        baseline = _safe_solve(system, weights * y)
        residual = y - baseline
        negative = residual[residual < 0]
        if negative.size < 2:
            break
        mean = float(np.mean(negative))
        deviation = float(np.std(negative))
        if deviation <= np.finfo(float).eps:
            break
        exponent = np.clip(2.0 * (residual - (2.0 * deviation - mean)) / deviation, -700.0, 700.0)
        new_weights = 1.0 / (1.0 + np.exp(exponent))
        relative_change = np.linalg.norm(weights - new_weights) / max(np.linalg.norm(weights), np.finfo(float).eps)
        weights = new_weights
        if relative_change <= settings.tolerance:
            break
    return baseline


def _safe_solve(matrix: sparse.csc_matrix, rhs: np.ndarray) -> np.ndarray:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=MatrixRankWarning)
        try:
            return np.asarray(spsolve(matrix, rhs), dtype=np.float64)
        except (MatrixRankWarning, RuntimeError, ValueError) as exc:
            raise BaselineError(f"Sparse baseline solve failed: {exc}") from exc


def _validate(y: np.ndarray, settings: BaselineSettings) -> None:
    if y.ndim != 1 or y.size < 3:
        raise BaselineError("Baseline input must be a one-dimensional array with at least three points")
    if not np.isfinite(y).all():
        raise BaselineError("Baseline input contains NaN or infinite values")
    if settings.lam <= 0:
        raise BaselineError("Lambda must be greater than zero")
    if settings.algorithm == "als" and not 0 < settings.asymmetry < 1:
        raise BaselineError("Asymmetry must be between zero and one")
    if settings.max_iterations < 1:
        raise BaselineError("Maximum iterations must be at least one")
