from __future__ import annotations

import numpy as np
from matplotlib.colors import FuncNorm
from matplotlib.figure import Figure

from .config import PlotSettings, RoiSettings
from .processing import ProcessingResult


ROI_FALLBACK_COLORS = ("#2f8bc2", "#e07a2f", "#36a269", "#8c65b8", "#c94c61")


def create_result_figure(
    result: ProcessingResult,
    settings: PlotSettings,
    roi_settings: RoiSettings | None = None,
) -> Figure:
    include_rois = bool(roi_settings and roi_settings.enabled and result.roi_values.size)
    figure = Figure(figsize=(settings.width_inches, settings.height_inches), dpi=settings.dpi, constrained_layout=True)
    if include_rois:
        intensity_axis, roi_axis = figure.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [4, 1]})
    else:
        intensity_axis = figure.subplots()
        roi_axis = None

    image_data = result.final.copy()
    if settings.flip_x_data:
        image_data = np.flip(image_data, axis=1)
    if settings.flip_y_data:
        image_data = np.flip(image_data, axis=0)

    y_start, y_stop, series_positions, y_label = series_axis_geometry(settings, result.curve_count)
    extent = [float(result.x[0]), float(result.x[-1]), y_start, y_stop]
    image_data, kwargs, colorbar_label = prepare_heatmap_display(image_data, settings)
    image = intensity_axis.imshow(
        image_data,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=settings.colormap,
        **kwargs,
    )
    intensity_axis.set_xlabel(settings.x_label)
    intensity_axis.set_ylabel(y_label)
    _draw_roi_ranges(intensity_axis, roi_settings)
    # Keep the image flush with the requested spectral bounds. ROI spans and
    # annotations must not make Matplotlib expand the horizontal axis.
    intensity_axis.set_xlim(extent[0], extent[1])
    if settings.flip_x_axis:
        intensity_axis.invert_xaxis()
    if settings.flip_y_axis:
        intensity_axis.invert_yaxis()
    if settings.colorbar:
        figure.colorbar(image, ax=intensity_axis, label=colorbar_label)

    if roi_axis is not None and roi_settings is not None:
        times = series_positions
        if settings.flip_y_data:
            times = np.flip(times)
        for index, (values, name) in enumerate(zip(result.roi_values, result.roi_names)):
            color = roi_settings.ranges[index].color.strip() or None
            roi_axis.plot(values, times, marker="o", markersize=2.5, linewidth=1.0, label=name, color=color)
        roi_axis.set_xlabel("ROI")
        roi_axis.tick_params(axis="y", labelleft=False)
        if len(result.roi_names) <= 8:
            roi_axis.legend(fontsize="small")
    return figure


def prepare_heatmap_display(
    image_data: np.ndarray,
    settings: PlotSettings,
) -> tuple[np.ndarray, dict[str, object], str]:
    """Prepare heatmap colors without changing the processed numerical result.

    Linear and robust modes keep raw intensity values. HDR uses a shifted asinh
    normalization so the colorbar can still report raw intensity. Per-scan mode
    deliberately converts each displayed row to a relative 0..1 contrast scale.
    """
    data = np.asarray(image_data, dtype=np.float64)
    mode = settings.display_scaling
    if mode == "linear":
        kwargs: dict[str, object] = {}
        if settings.value_limits_enabled:
            kwargs.update(vmin=settings.value_min, vmax=settings.value_max)
        return data, kwargs, "Intensity"

    low, high = _finite_percentile_bounds(
        data,
        settings.percentile_low,
        settings.percentile_high,
    )
    if mode == "robust":
        return data, {"vmin": low, "vmax": high}, "Intensity"

    if mode == "asinh":
        # Put the lower robust bound at zero, then retain a small linear region
        # before progressively compressing large peaks. FuncNorm supplies the
        # inverse transform, so colorbar tick labels remain in raw intensity.
        linear_width = max((high - low) * 0.01, np.finfo(np.float64).eps)

        def forward(values):
            return np.arcsinh((np.asarray(values, dtype=np.float64) - low) / linear_width)

        def inverse(values):
            return low + linear_width * np.sinh(np.asarray(values, dtype=np.float64))

        norm = FuncNorm((forward, inverse), vmin=low, vmax=high, clip=True)
        return data, {"norm": norm}, "Intensity (asinh/HDR display)"

    if mode == "per_scan":
        normalized = np.full_like(data, np.nan, dtype=np.float64)
        rows = data.reshape(1, -1) if data.ndim == 1 else data
        normalized_rows = normalized.reshape(1, -1) if normalized.ndim == 1 else normalized
        for row_index, row in enumerate(rows):
            finite_mask = np.isfinite(row)
            if not np.any(finite_mask):
                continue
            row_low, row_high = _finite_percentile_bounds(
                row[finite_mask],
                settings.percentile_low,
                settings.percentile_high,
            )
            normalized_rows[row_index, finite_mask] = np.clip(
                (row[finite_mask] - row_low) / (row_high - row_low),
                0.0,
                1.0,
            )
        return normalized, {"vmin": 0.0, "vmax": 1.0}, "Relative intensity per scan"

    raise ValueError(f"Unknown heatmap display scaling: {mode}")


def _finite_percentile_bounds(
    values: np.ndarray,
    percentile_low: float,
    percentile_high: float,
) -> tuple[float, float]:
    if not 0.0 <= percentile_low < percentile_high <= 100.0:
        raise ValueError("Heatmap percentiles must satisfy 0 <= lower < upper <= 100")
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("Heatmap data contains no finite intensity values")
    low, high = np.percentile(finite, [percentile_low, percentile_high])
    low = float(low); high = float(high)
    if high <= low:
        padding = max(abs(low) * 1e-9, 1e-12)
        low -= padding
        high += padding
    return low, high


def series_axis_geometry(
    settings: PlotSettings,
    curve_count: int,
) -> tuple[float, float, np.ndarray, str]:
    """Return heatmap bounds, row centers, and the generated y-axis label."""
    if curve_count < 1:
        raise ValueError("A heatmap requires at least one curve")
    if settings.automatic_time_axis:
        if settings.scan_interval_seconds <= 0:
            raise ValueError("Time per scan must be greater than zero")
        minutes_per_scan = settings.scan_interval_seconds / 60.0
        positions = (np.arange(curve_count, dtype=np.float64) + 0.5) * minutes_per_scan
        return 0.0, curve_count * minutes_per_scan, positions, "Time / min"
    positions = np.arange(1, curve_count + 1, dtype=np.float64)
    return 0.5, curve_count + 0.5, positions, "Scan number"


def draw_curve_preview(
    figure: Figure,
    x: np.ndarray,
    original: np.ndarray,
    baseline: np.ndarray | None = None,
    corrected: np.ndarray | None = None,
    final: np.ndarray | None = None,
    title: str = "",
    x_label: str = "x",
    roi_settings: RoiSettings | None = None,
    baseline_label: str = "Baseline",
    reference: np.ndarray | None = None,
) -> None:
    figure.clear()
    axis = figure.subplots()
    axis.plot(x, original, label="Original", linewidth=1.0)
    if reference is not None:
        axis.plot(x, reference, label="Scaled JAR/reference", color="tab:purple", linewidth=1.2)
    if baseline is not None:
        axis.plot(x, baseline, label=baseline_label, linewidth=1.2)
    if corrected is not None:
        axis.plot(x, corrected, label="Corrected", linewidth=1.0)
    if final is not None and corrected is not None and not np.allclose(final, corrected):
        axis.plot(x, final, label="Final", linewidth=1.0, alpha=0.85)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Intensity")
    _draw_roi_ranges(axis, roi_settings)
    finite_x = np.asarray(x, dtype=np.float64)
    if finite_x.size and np.isfinite(finite_x).all():
        low = float(np.min(finite_x)); high = float(np.max(finite_x))
        if low < high:
            axis.set_xlim(low, high)
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()


def _draw_roi_ranges(axis, roi_settings: RoiSettings | None) -> None:
    if roi_settings is None or not roi_settings.enabled:
        return
    for index, roi in enumerate(roi_settings.ranges):
        low, high = sorted((roi.start, roi.stop))
        color = roi.color.strip() or ROI_FALLBACK_COLORS[index % len(ROI_FALLBACK_COLORS)]
        axis.axvspan(low, high, color=color, alpha=0.16, linewidth=0)
        axis.text(
            (low + high) / 2.0,
            0.985,
            roi.name or f"ROI {index + 1}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize="x-small",
            color=color,
        )
