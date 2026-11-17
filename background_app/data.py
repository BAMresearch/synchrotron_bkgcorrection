from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class DataValidationError(ValueError):
    """Raised when imported spectral data cannot be processed safely."""


@dataclass(slots=True)
class LoadedFile:
    path: Path
    x: np.ndarray
    ys: np.ndarray
    header: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    detected_delimiter: str = "whitespace"
    data_type: str = "generic"
    x_label: str = ""

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.x = np.asarray(self.x, dtype=np.float64)
        self.ys = np.asarray(self.ys, dtype=np.float64)
        if self.ys.ndim == 1:
            self.ys = self.ys.reshape(1, -1)
        if self.x.ndim != 1 or self.ys.ndim != 2:
            raise DataValidationError(f"{self.path.name}: expected a one-dimensional x-axis and a two-dimensional intensity array")
        if self.x.size < 3:
            raise DataValidationError(f"{self.path.name}: at least three data points are required")
        if self.ys.shape[1] != self.x.size:
            raise DataValidationError(f"{self.path.name}: x-axis and intensity lengths differ")
        if not np.isfinite(self.x).all() or not np.isfinite(self.ys).all():
            raise DataValidationError(f"{self.path.name}: NaN or infinite values are not supported")
        if not self.labels:
            self.labels = [f"{self.path.name}[{index + 1}]" for index in range(self.ys.shape[0])]

    @property
    def points(self) -> int:
        return int(self.x.size)

    @property
    def curves(self) -> int:
        return int(self.ys.shape[0])

    @property
    def x_min(self) -> float:
        return float(np.min(self.x))

    @property
    def x_max(self) -> float:
        return float(np.max(self.x))


@dataclass(slots=True)
class DatasetData:
    x: np.ndarray
    ys: np.ndarray
    labels: list[str]
    source_paths: list[Path]


def combine_files(files: list[LoadedFile], alignment: str = "strict") -> DatasetData:
    if not files:
        raise DataValidationError("No input files have been loaded")
    target_x = np.asarray(files[0].x, dtype=np.float64)
    all_ys: list[np.ndarray] = []
    labels: list[str] = []

    for item in files:
        if item.x.shape == target_x.shape and np.allclose(item.x, target_x, rtol=1e-9, atol=1e-12):
            aligned = item.ys
        elif alignment == "interpolate":
            aligned = _interpolate_rows(item.x, item.ys, target_x, item.path.name)
        else:
            raise DataValidationError(
                f"{item.path.name}: x-axis does not match {files[0].path.name}. "
                "Enable interpolation or remove the mismatching file."
            )
        all_ys.append(aligned)
        labels.extend(item.labels)

    return DatasetData(
        x=target_x.copy(),
        ys=np.vstack(all_ys),
        labels=labels,
        source_paths=list(dict.fromkeys(item.path for item in files)),
    )


def _interpolate_rows(source_x: np.ndarray, source_ys: np.ndarray, target_x: np.ndarray, name: str) -> np.ndarray:
    source_order = np.argsort(source_x)
    target_order = np.argsort(target_x)
    source_sorted = source_x[source_order]
    target_sorted = target_x[target_order]
    if target_sorted[0] < source_sorted[0] or target_sorted[-1] > source_sorted[-1]:
        raise DataValidationError(f"{name}: interpolation would require extrapolation")
    interpolated = np.empty((source_ys.shape[0], target_x.size), dtype=np.float64)
    for index, row in enumerate(source_ys):
        sorted_result = np.interp(target_sorted, source_sorted, row[source_order])
        interpolated[index, target_order] = sorted_result
    return interpolated
