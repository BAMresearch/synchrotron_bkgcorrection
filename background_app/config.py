from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


@dataclass(slots=True)
class ImportSettings:
    header_rows: int = 0
    auto_header: bool = True
    delimiter: str = "auto"
    alignment: str = "strict"
    recursive_folders: bool = True


@dataclass(slots=True)
class BaselineSettings:
    enabled: bool = True
    algorithm: str = "als"
    lam: float = 2000.0
    asymmetry: float = 0.02
    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass(slots=True)
class NormalizationSettings:
    mode: str = "area"
    ground: bool = False


@dataclass(slots=True)
class DataSettings:
    data_type: str = "auto"
    range_enabled: bool = False
    range_start: float = 2.0
    range_stop: float = 47.0
    input_unit: str = "unchanged"
    output_unit: str = "unchanged"
    wavelength_angstrom: float = 1.5406


@dataclass(slots=True)
class ReferenceSettings:
    enabled: bool = False
    filename: str = ""
    curve_index: int | None = None
    curve_label: str = ""
    fit_range_enabled: bool = False
    fit_range_start: float = 0.0
    fit_range_stop: float = 0.0


@dataclass(slots=True)
class RoiRange:
    start: float = 0.0
    stop: float = 1.0
    color: str = ""
    name: str = "ROI"


@dataclass(slots=True)
class RoiSettings:
    enabled: bool = False
    selection: str = "area"
    normalization: str = "total"
    ranges: list[RoiRange] = field(default_factory=list)


@dataclass(slots=True)
class PlotSettings:
    enabled: bool = True
    colormap: str = "hot"
    display_scaling: str = "robust"
    percentile_low: float = 1.0
    percentile_high: float = 99.5
    x_label: str = "q / nm⁻¹"
    automatic_time_axis: bool = True
    scan_interval_seconds: float = 30.0
    colorbar: bool = False
    flip_x_data: bool = False
    flip_y_data: bool = False
    flip_x_axis: bool = False
    flip_y_axis: bool = False
    value_limits_enabled: bool = False
    value_min: float = 0.0
    value_max: float = 1.0
    width_inches: float = 8.0
    height_inches: float = 5.0
    dpi: int = 160
    image_format: str = "png"


@dataclass(slots=True)
class RuntimeSettings:
    workers: int = 0  # 0 = adaptive automatic selection


@dataclass(slots=True)
class OutputSettings:
    directory: str = ""
    delimiter: str = "\t"
    save_corrected: bool = True
    save_baseline: bool = True
    save_original: bool = False
    save_reference_corrected: bool = False
    save_plot: bool = True
    save_rois: bool = True


@dataclass(slots=True)
class AppSettings:
    schema_version: int = SCHEMA_VERSION
    import_options: ImportSettings = field(default_factory=ImportSettings)
    baseline: BaselineSettings = field(default_factory=BaselineSettings)
    normalization: NormalizationSettings = field(default_factory=NormalizationSettings)
    data: DataSettings = field(default_factory=DataSettings)
    reference: ReferenceSettings = field(default_factory=ReferenceSettings)
    rois: RoiSettings = field(default_factory=RoiSettings)
    plot: PlotSettings = field(default_factory=PlotSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, filename: str | Path) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, filename: str | Path) -> "AppSettings":
        raw = json.loads(Path(filename).read_text(encoding="utf-8"))
        if "schema_version" not in raw and "baseline" in raw and "io" in raw:
            return cls.from_legacy_dict(raw)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppSettings":
        def section(name: str) -> dict[str, Any]:
            value = raw.get(name, {})
            return value if isinstance(value, dict) else {}

        roi_raw = section("rois")
        roi_ranges = []
        for item in roi_raw.get("ranges", []):
            if isinstance(item, dict):
                roi_ranges.append(RoiRange(**_known_kwargs(RoiRange, item)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                roi_ranges.append(RoiRange(float(item[0]), float(item[1]), str(item[2]) if len(item) > 2 else ""))

        rois = RoiSettings(**_known_kwargs(RoiSettings, {**roi_raw, "ranges": roi_ranges}))
        plot_raw = section("plot")
        # Earlier app settings stored a free-text label and a y-axis step.
        # Convert them into the explicit seconds-per-scan model when imported.
        if "scan_interval_seconds" not in plot_raw and "time_step" in plot_raw:
            old_step = float(plot_raw.get("time_step", 1.0))
            old_label = str(plot_raw.get("y_label", "Time / s")).lower()
            multiplier = 3600.0 if "/ h" in old_label else 60.0 if "/ min" in old_label else 1.0
            plot_raw = {
                **plot_raw,
                "automatic_time_axis": old_label.strip().startswith("time"),
                "scan_interval_seconds": old_step * multiplier,
            }
        return cls(
            schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
            import_options=ImportSettings(**_known_kwargs(ImportSettings, section("import_options"))),
            baseline=BaselineSettings(**_known_kwargs(BaselineSettings, section("baseline"))),
            normalization=NormalizationSettings(**_known_kwargs(NormalizationSettings, section("normalization"))),
            data=DataSettings(**_known_kwargs(DataSettings, section("data"))),
            reference=ReferenceSettings(**_known_kwargs(ReferenceSettings, section("reference"))),
            rois=rois,
            plot=PlotSettings(**_known_kwargs(PlotSettings, plot_raw)),
            runtime=RuntimeSettings(**_known_kwargs(RuntimeSettings, section("runtime"))),
            output=OutputSettings(**_known_kwargs(OutputSettings, section("output"))),
        )

    @classmethod
    def from_legacy_dict(cls, raw: dict[str, Any]) -> "AppSettings":
        settings = cls()
        parallel = raw.get("parallel", {})
        io = raw.get("io", {})
        data = raw.get("data", {})
        baseline = raw.get("baseline", {})
        normalization = raw.get("normalization", {})
        jar = raw.get("jar", {})
        rois = raw.get("rois", {})
        plot = raw.get("plot", {})

        settings.runtime.workers = 0 if parallel.get("cores", "auto") == "auto" else int(parallel["cores"])
        if not parallel.get("enable", True):
            settings.runtime.workers = 1
        settings.output.directory = str(io.get("out_dir", settings.output.directory))
        settings.output.delimiter = str(io.get("dat_file_sep", settings.output.delimiter))
        settings.import_options.header_rows = int(io.get("head_row_count", 0))
        settings.import_options.auto_header = False

        settings.data.range_start = float(data.get("range_start", settings.data.range_start))
        settings.data.range_stop = float(data.get("range_stop", settings.data.range_stop))
        # The legacy file had no separate enable switch. Keep its limits as
        # editable suggestions, but do not silently crop newly imported data.
        settings.data.range_enabled = False
        unit_map = {0: "q_nm", 1: "q_angstrom", 2: "two_theta_deg"}
        unit_in = int(data.get("unit_in", -1))
        unit_out = int(data.get("unit_out", -1))
        # Equal legacy units performed no conversion. Preserve that behavior without
        # claiming a physical unit that may conflict with the old custom plot label.
        if unit_in == unit_out:
            settings.data.input_unit = settings.data.output_unit = "unchanged"
        else:
            settings.data.input_unit = unit_map.get(unit_in, "unchanged")
            settings.data.output_unit = unit_map.get(unit_out, "unchanged")

        settings.baseline.enabled = bool(baseline.get("enable", True))
        settings.baseline.algorithm = {0: "arpls", 1: "als"}.get(int(baseline.get("algorithm", 1)), "als")
        settings.baseline.max_iterations = int(baseline.get("itermax", settings.baseline.max_iterations))
        settings.baseline.lam = float(baseline.get("lam", settings.baseline.lam))
        settings.baseline.asymmetry = float(baseline.get("ratio", settings.baseline.asymmetry))
        settings.baseline.tolerance = 0.0 if settings.baseline.algorithm == "als" else settings.baseline.asymmetry

        for mode in ("area", "sum", "max"):
            if normalization.get(mode, False):
                settings.normalization.mode = mode
                break
        else:
            settings.normalization.mode = "none"
        settings.normalization.ground = bool(normalization.get("ground", False))

        settings.reference.enabled = bool(jar.get("enable", False))
        settings.reference.fit_range_start = float(jar.get("range_start", 0.0))
        settings.reference.fit_range_stop = float(jar.get("range_stop", 0.0))
        settings.reference.fit_range_enabled = settings.reference.enabled

        settings.rois.enabled = bool(rois.get("enable", False))
        settings.rois.selection = "max" if rois.get("selection", {}).get("max", False) else "area"
        roi_normalize = rois.get("normalize", {})
        settings.rois.normalization = (
            "total" if roi_normalize.get("sum", False) else
            "linear" if roi_normalize.get("sum_linear", False) else
            "max" if roi_normalize.get("max", False) else "none"
        )
        settings.rois.ranges = [
            RoiRange(float(item[0]), float(item[1]), str(item[2]) if len(item) > 2 else "", f"ROI {index + 1}")
            for index, item in enumerate(rois.get("ranges", [])) if len(item) >= 2
        ]

        settings.plot.enabled = bool(plot.get("enable", True))
        legacy_time_step = float(plot.get("time_step", 1.0))
        legacy_time_unit = str(plot.get("y_unit", "s")).strip().lower()
        multiplier = 3600.0 if legacy_time_unit in {"h", "hr", "hour", "hours"} else 60.0 if legacy_time_unit in {"m", "min", "minute", "minutes"} else 1.0
        settings.plot.automatic_time_axis = True
        settings.plot.scan_interval_seconds = legacy_time_step * multiplier
        settings.plot.colormap = str(plot.get("heatmap", "hot"))
        settings.plot.colorbar = bool(plot.get("colorbar", False))
        settings.plot.flip_x_data = bool(plot.get("flip_x_data", False))
        settings.plot.flip_y_data = bool(plot.get("flip_y_data", False))
        settings.plot.flip_x_axis = bool(plot.get("flip_x_ticks", False))
        settings.plot.flip_y_axis = bool(plot.get("flip_y_ticks", False))
        settings.plot.x_label = str(plot.get("x_unit", settings.plot.x_label))
        return settings


def _known_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    names = cls.__dataclass_fields__.keys()
    return {key: value for key, value in values.items() if key in names}
