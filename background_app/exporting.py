from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import AppSettings
from .plotting import create_result_figure, series_axis_geometry
from .processing import ProcessingResult


@dataclass(slots=True)
class ExportManifest:
    output_directory: Path
    files: list[Path] = field(default_factory=list)


def export_result(
    result: ProcessingResult,
    settings: AppSettings,
    output_directory: str | Path,
    dataset_name: str,
) -> ExportManifest:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = safe_name(dataset_name) or "dataset"
    manifest = ExportManifest(output)
    metadata = {
        "application_schema": settings.schema_version,
        "source_files": result.source_paths,
        "elapsed_seconds": result.elapsed_seconds,
        "workers_used": result.workers_used,
        "settings": settings.to_dict(),
    }

    if settings.output.save_corrected:
        manifest.files.append(_write_matrix(output / f"{stem}_corrected.dat", result.x, result.final, result.labels, settings.output.delimiter, metadata))
    if settings.output.save_baseline:
        manifest.files.append(_write_matrix(output / f"{stem}_baseline.dat", result.x, result.baselines, result.labels, settings.output.delimiter, metadata))
    if settings.output.save_original:
        manifest.files.append(_write_matrix(output / f"{stem}_original.dat", result.x, result.original, result.labels, settings.output.delimiter, metadata))
    if settings.output.save_reference_corrected and np.any(np.isfinite(result.reference_scales)):
        manifest.files.append(_write_matrix(output / f"{stem}_reference_corrected.dat", result.x, result.reference_corrected, result.labels, settings.output.delimiter, metadata))

    if settings.output.save_rois and result.roi_values.size:
        roi_path = output / f"{stem}_rois.csv"
        _, _, series_positions, _ = series_axis_geometry(settings.plot, result.curve_count)
        roi_data = np.column_stack([series_positions, result.roi_values.T])
        coordinate_header = "time_min" if settings.plot.automatic_time_axis else "scan_number"
        header = ",".join([coordinate_header, *[_clean_label(name) for name in result.roi_names]])
        with roi_path.open("w", encoding="utf-8", newline="\n") as stream:
            np.savetxt(stream, roi_data, delimiter=",", header=header, comments="", fmt="%.10g")
        manifest.files.append(roi_path)

    if settings.plot.enabled and settings.output.save_plot:
        plot_path = output / f"{stem}.{settings.plot.image_format}"
        figure = create_result_figure(result, settings.plot, settings.rois)
        figure.savefig(plot_path, dpi=settings.plot.dpi, format=settings.plot.image_format)
        manifest.files.append(plot_path)

    manifest_path = output / f"{stem}_run.json"
    manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest.files.append(manifest_path)
    return manifest


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")


def _write_matrix(
    path: Path,
    x: np.ndarray,
    rows: np.ndarray,
    labels: list[str],
    delimiter: str,
    metadata: dict,
) -> Path:
    data = np.column_stack([x, rows.T])
    compact_metadata = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    label_line = delimiter.join(["x", *[_clean_label(label) for label in labels]])
    header = f"Background Correction App\nmetadata={compact_metadata}\n{label_line}"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        np.savetxt(stream, data, delimiter=delimiter, header=header, comments="# ", fmt="%.10e")
    return path


def _clean_label(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ").replace(",", ";")
