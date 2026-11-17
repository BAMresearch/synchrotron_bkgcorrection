from __future__ import annotations

import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .baseline import fit_baseline
from .config import AppSettings, BaselineSettings, NormalizationSettings, RoiSettings
from .data import DataValidationError, DatasetData, LoadedFile


ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(slots=True)
class PreparedReference:
    raw: np.ndarray
    corrected: np.ndarray
    fit_mask: np.ndarray


@dataclass(slots=True)
class CurveResult:
    original: np.ndarray
    reference_corrected: np.ndarray
    reference_scaled: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    grounded: np.ndarray
    final: np.ndarray
    reference_scale: float


@dataclass(slots=True)
class ProcessingResult:
    x: np.ndarray
    labels: list[str]
    source_paths: list[str]
    original: np.ndarray
    reference_corrected: np.ndarray
    reference_scaled: np.ndarray
    baselines: np.ndarray
    corrected: np.ndarray
    grounded: np.ndarray
    final: np.ndarray
    reference_scales: np.ndarray
    roi_values: np.ndarray
    roi_names: list[str]
    elapsed_seconds: float
    workers_used: int

    @property
    def curve_count(self) -> int:
        return int(self.final.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.final.shape[1])

    @property
    def total_subtracted(self) -> np.ndarray:
        """Return everything removed before grounding and normalization.

        The fitted baseline is calculated after reference/JAR subtraction, so it
        cannot be compared directly with the untouched original signal. This
        cumulative curve keeps previews on one common intensity scale.
        """
        return self.original - self.corrected


def process_dataset(
    dataset: DatasetData,
    settings: AppSettings,
    reference_file: LoadedFile | None = None,
    progress: ProgressCallback | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ProcessingResult:
    started = time.perf_counter()
    _notify(progress, "prepare", 0, dataset.ys.shape[0], "Preparing data")

    x_converted = convert_axis(
        dataset.x,
        settings.data.input_unit,
        settings.data.output_unit,
        settings.data.wavelength_angstrom,
    )
    selection = np.ones(x_converted.size, dtype=bool)
    if settings.data.range_enabled:
        low, high = sorted((settings.data.range_start, settings.data.range_stop))
        selection = (x_converted >= low) & (x_converted <= high)
    if np.count_nonzero(selection) < 3:
        raise DataValidationError("The selected x-range contains fewer than three points")

    x = x_converted[selection]
    ys = np.asarray(dataset.ys[:, selection], dtype=np.float64)
    reference = _prepare_reference(x, settings, reference_file)
    workers = resolve_worker_count(settings.runtime.workers, ys.shape[0], ys.shape[1])
    results: list[CurveResult | None] = [None] * ys.shape[0]

    if workers == 1:
        _process_sequential(x, ys, settings, reference, results, progress, cancelled, dataset.labels)
    else:
        try:
            _process_parallel(x, ys, settings, reference, workers, results, progress, cancelled, dataset.labels)
        except (OSError, BrokenProcessPool) as exc:
            workers = 1
            results = [None] * ys.shape[0]
            _notify(progress, "prepare", 0, ys.shape[0], f"Parallel processing unavailable ({exc}); using one worker")
            _process_sequential(x, ys, settings, reference, results, progress, cancelled, dataset.labels)

    completed = [item for item in results if item is not None]
    if len(completed) != ys.shape[0]:
        raise RuntimeError("Processing ended before all curves were completed")

    final = np.vstack([item.final for item in completed])
    roi_values, roi_names = calculate_rois(x, final, settings.rois)
    elapsed = time.perf_counter() - started
    _notify(progress, "complete", ys.shape[0], ys.shape[0], f"Finished in {elapsed:.2f} s")
    return ProcessingResult(
        x=x,
        labels=list(dataset.labels),
        source_paths=[str(path) for path in dataset.source_paths],
        original=np.vstack([item.original for item in completed]),
        reference_corrected=np.vstack([item.reference_corrected for item in completed]),
        reference_scaled=np.vstack([item.reference_scaled for item in completed]),
        baselines=np.vstack([item.baseline for item in completed]),
        corrected=np.vstack([item.corrected for item in completed]),
        grounded=np.vstack([item.grounded for item in completed]),
        final=final,
        reference_scales=np.asarray([item.reference_scale for item in completed]),
        roi_values=roi_values,
        roi_names=roi_names,
        elapsed_seconds=elapsed,
        workers_used=workers,
    )


def resolve_worker_count(requested: int, curves: int, points: int) -> int:
    if curves < 1:
        return 1
    logical_cpus = max(1, os.cpu_count() or 1)
    if requested > 0:
        return max(1, min(requested, curves, logical_cpus))
    # Spawning processes for a handful of short spectra is slower than processing locally.
    if curves < 12 or curves * points < 6000:
        return 1
    automatic_limit = max(1, logical_cpus - 1)
    return max(1, min(automatic_limit, curves, 8))


def convert_axis(values: np.ndarray, input_unit: str, output_unit: str, wavelength_angstrom: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if input_unit == output_unit or input_unit == "unchanged" or output_unit == "unchanged":
        return x.copy()
    if wavelength_angstrom <= 0:
        raise DataValidationError("Wavelength must be greater than zero")

    if input_unit == "q_nm":
        q_angstrom = x / 10.0
    elif input_unit == "q_angstrom":
        q_angstrom = x
    elif input_unit == "two_theta_deg":
        q_angstrom = 4.0 * np.pi * np.sin(np.deg2rad(x) / 2.0) / wavelength_angstrom
    else:
        raise DataValidationError(f"Unknown input unit: {input_unit}")

    if output_unit == "q_nm":
        return q_angstrom * 10.0
    if output_unit == "q_angstrom":
        return q_angstrom.copy()
    if output_unit == "two_theta_deg":
        argument = wavelength_angstrom * q_angstrom / (4.0 * np.pi)
        if np.any(np.abs(argument) > 1.0):
            raise DataValidationError("q values are outside the selected wavelength's 2θ domain")
        return np.rad2deg(2.0 * np.arcsin(argument))
    raise DataValidationError(f"Unknown output unit: {output_unit}")


def calculate_rois(x: np.ndarray, final: np.ndarray, settings: RoiSettings) -> tuple[np.ndarray, list[str]]:
    if not settings.enabled or not settings.ranges:
        return np.empty((0, final.shape[0]), dtype=np.float64), []
    order = np.argsort(x)
    sorted_x = x[order]
    rows: list[np.ndarray] = []
    names: list[str] = []
    for index, roi in enumerate(settings.ranges):
        low, high = sorted((roi.start, roi.stop))
        mask = (sorted_x >= low) & (sorted_x <= high)
        if not np.any(mask):
            raise DataValidationError(f"{roi.name or f'ROI {index + 1}'} contains no data points")
        selected = final[:, order][:, mask]
        if settings.selection == "max":
            values = np.max(selected, axis=1)
        else:
            values = np.trapezoid(selected, x=sorted_x[mask], axis=1)
        rows.append(values)
        names.append(roi.name or f"ROI {index + 1}")
    result = np.vstack(rows)

    epsilon = np.finfo(float).eps
    if settings.normalization == "total":
        totals = np.sum(result, axis=0)
        result = np.divide(result, totals, out=np.zeros_like(result), where=np.abs(totals) > epsilon)
    elif settings.normalization == "linear":
        mean_total = float(np.mean(np.sum(result, axis=0)))
        if abs(mean_total) <= epsilon:
            raise DataValidationError("ROI linear normalization has a zero mean total")
        result = result / mean_total
    elif settings.normalization == "max":
        maximum = float(np.max(np.abs(result)))
        if maximum <= epsilon:
            raise DataValidationError("ROI maximum normalization has a zero maximum")
        result = result / maximum
    return result, names


def _prepare_reference(x: np.ndarray, settings: AppSettings, reference_file: LoadedFile | None) -> PreparedReference | None:
    if not settings.reference.enabled:
        return None
    if reference_file is None:
        raise DataValidationError("Reference correction is enabled, but no reference file is loaded")

    reference_x = convert_axis(
        reference_file.x,
        settings.data.input_unit,
        settings.data.output_unit,
        settings.data.wavelength_angstrom,
    )
    order = np.argsort(reference_x)
    sorted_x = reference_x[order]
    target_order = np.argsort(x)
    sorted_target = x[target_order]
    if sorted_target[0] < sorted_x[0] or sorted_target[-1] > sorted_x[-1]:
        raise DataValidationError("The reference spectrum does not cover the complete selected x-range")
    sorted_reference = np.interp(sorted_target, sorted_x, reference_file.ys[0][order])
    raw = np.empty_like(x)
    raw[target_order] = sorted_reference
    baseline = fit_baseline(raw, settings.baseline)
    corrected = raw - baseline

    fit_mask = np.ones(x.size, dtype=bool)
    if settings.reference.fit_range_enabled:
        low, high = sorted((settings.reference.fit_range_start, settings.reference.fit_range_stop))
        fit_mask = (x >= low) & (x <= high)
    if np.count_nonzero(fit_mask) < 2:
        raise DataValidationError("The reference fit range contains fewer than two points")
    if np.linalg.norm(corrected[fit_mask]) <= np.finfo(float).eps:
        raise DataValidationError("The corrected reference signal is zero inside the fit range")
    return PreparedReference(raw=raw, corrected=corrected, fit_mask=fit_mask)


def _process_curve(
    x: np.ndarray,
    y: np.ndarray,
    baseline_settings: BaselineSettings,
    normalization_settings: NormalizationSettings,
    reference: PreparedReference | None,
    cancelled: Callable[[], bool] | None = None,
) -> CurveResult:
    original = np.asarray(y, dtype=np.float64)
    reference_corrected = original.copy()
    reference_scaled = np.zeros_like(original)
    reference_scale = float("nan")

    if reference is not None:
        scaling_baseline = fit_baseline(original, baseline_settings, cancelled)
        scaling_signal = original - scaling_baseline
        ref = reference.corrected[reference.fit_mask]
        target = scaling_signal[reference.fit_mask]
        denominator = float(np.dot(ref, ref))
        reference_scale = float(np.dot(ref, target) / denominator)
        reference_scaled = reference_scale * reference.raw
        reference_corrected = original - reference_scaled

    if baseline_settings.enabled:
        baseline = fit_baseline(reference_corrected, baseline_settings, cancelled)
        corrected = reference_corrected - baseline
    else:
        baseline = np.zeros_like(original)
        corrected = reference_corrected.copy()

    grounded = corrected - np.min(corrected) if normalization_settings.ground else corrected.copy()
    final = _normalize(x, grounded, normalization_settings.mode)
    return CurveResult(
        original=original.copy(),
        reference_corrected=reference_corrected,
        reference_scaled=reference_scaled,
        baseline=baseline,
        corrected=corrected,
        grounded=grounded,
        final=final,
        reference_scale=reference_scale,
    )


def _normalize(x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y.copy()
    if mode == "area":
        denominator = abs(float(np.trapezoid(y, x=x)))
    elif mode == "sum":
        denominator = abs(float(np.sum(y)))
    elif mode == "max":
        denominator = float(np.max(np.abs(y)))
    else:
        raise DataValidationError(f"Unknown normalization mode: {mode}")
    if denominator <= np.finfo(float).eps:
        raise DataValidationError(f"Cannot apply {mode} normalization to a zero signal")
    return y / denominator


def _process_parallel(
    x: np.ndarray,
    ys: np.ndarray,
    settings: AppSettings,
    reference: PreparedReference | None,
    workers: int,
    output: list[CurveResult | None],
    progress: ProgressCallback | None,
    cancelled: Callable[[], bool] | None,
    labels: list[str],
) -> None:
    chunk_count = min(ys.shape[0], workers * 2)
    chunks = [chunk.tolist() for chunk in np.array_split(np.arange(ys.shape[0]), chunk_count) if chunk.size]
    context = multiprocessing.get_context("spawn")
    completed = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                _process_batch,
                indices,
                x,
                ys[indices],
                settings.baseline,
                settings.normalization,
                reference,
            ): indices
            for indices in chunks
        }
        for future in as_completed(futures):
            if cancelled and cancelled():
                for pending in futures:
                    pending.cancel()
                raise InterruptedError("Processing cancelled")
            for index, result in future.result():
                output[index] = result
            completed += len(futures[future])
            last_index = futures[future][-1]
            _notify(progress, "process", completed, ys.shape[0], labels[last_index])


def _process_sequential(
    x: np.ndarray,
    ys: np.ndarray,
    settings: AppSettings,
    reference: PreparedReference | None,
    output: list[CurveResult | None],
    progress: ProgressCallback | None,
    cancelled: Callable[[], bool] | None,
    labels: list[str],
) -> None:
    for index, y in enumerate(ys):
        if cancelled and cancelled():
            raise InterruptedError("Processing cancelled")
        output[index] = _process_curve(
            x, y, settings.baseline, settings.normalization, reference, cancelled
        )
        _notify(progress, "process", index + 1, ys.shape[0], labels[index])


def _process_batch(
    indices: list[int],
    x: np.ndarray,
    rows: np.ndarray,
    baseline_settings: BaselineSettings,
    normalization_settings: NormalizationSettings,
    reference: PreparedReference | None,
) -> list[tuple[int, CurveResult]]:
    return [
        (index, _process_curve(x, row, baseline_settings, normalization_settings, reference))
        for index, row in zip(indices, rows)
    ]


def _notify(callback: ProgressCallback | None, stage: str, completed: int, total: int, message: str) -> None:
    if callback:
        callback(stage, completed, total, message)
