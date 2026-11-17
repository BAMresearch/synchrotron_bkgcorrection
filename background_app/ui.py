from __future__ import annotations

import json
import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from .config import AppSettings, RoiRange
from .data import DataValidationError, DatasetData, LoadedFile, combine_files
from .exporting import ExportManifest, export_result
from .io import SUPPORTED_SUFFIXES, discover_files, load_file
from .plotting import ROI_FALLBACK_COLORS, create_result_figure, draw_curve_preview
from .processing import ProcessingResult, process_dataset


class CompactDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """A precise spin box that does not pad displayed values with trailing zeros."""

    def textFromValue(self, value: float) -> str:
        locale = QtCore.QLocale(self.locale())
        locale.setNumberOptions(locale.numberOptions() | QtCore.QLocale.NumberOption.OmitGroupSeparator)
        text = locale.toString(value, "f", self.decimals())
        decimal_point = locale.decimalPoint()
        if decimal_point in text:
            text = text.rstrip("0").rstrip(decimal_point)
        return text


_REFERENCE_ITEM_MIME = "application/x-background-correction-reference-item"


class ReorderTableWidget(QtWidgets.QTableWidget):
    """A row table whose internal drops are delegated to the backing file list."""

    rowsReordered = QtCore.Signal(int, int)

    def startDrag(self, supported_actions: QtCore.Qt.DropAction) -> None:
        items = self.selectedItems()
        if items:
            drag = QtGui.QDrag(self)
            drag.setMimeData(self.mimeData(items))
            # The backing list owns row moves, so Qt must keep the source cells.
            drag.exec(QtCore.Qt.DropAction.CopyAction)

    def mimeData(self, items: list[QtWidgets.QTableWidgetItem]) -> QtCore.QMimeData:
        mime_data = super().mimeData(items)
        rows = {item.row() for item in items}
        if len(rows) == 1:
            mime_data.setData(_REFERENCE_ITEM_MIME, str(next(iter(rows))).encode("ascii"))
        return mime_data

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.source() is not self or not event.mimeData().hasFormat(_REFERENCE_ITEM_MIME):
            super().dropEvent(event)
            return
        source = int(bytes(event.mimeData().data(_REFERENCE_ITEM_MIME)))
        target = self.rowAt(event.position().toPoint().y())
        if target < 0:
            target = self.rowCount()
        elif event.position().y() >= self.visualRect(self.model().index(target, 0)).center().y():
            target += 1
        if target > source:
            target -= 1
        target = max(0, min(target, self.rowCount() - 1))
        if target != source:
            self.rowsReordered.emit(source, target)
        event.setDropAction(QtCore.Qt.DropAction.CopyAction)
        event.accept()


class ReferenceDropLineEdit(QtWidgets.QLineEdit):
    """Reference field that accepts one spectrum from the data-series table."""

    itemDropped = QtCore.Signal(int)
    fileDropped = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        mime_data = event.mimeData()
        if mime_data.hasFormat(_REFERENCE_ITEM_MIME) or mime_data.hasUrls():
            event.setDropAction(QtCore.Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if event.mimeData().hasFormat(_REFERENCE_ITEM_MIME) or event.mimeData().hasUrls():
            event.setDropAction(QtCore.Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        mime_data = event.mimeData()
        if mime_data.hasFormat(_REFERENCE_ITEM_MIME):
            try:
                row = int(bytes(mime_data.data(_REFERENCE_ITEM_MIME)).decode("ascii"))
            except (TypeError, ValueError, UnicodeDecodeError):
                event.ignore()
                return
            self.itemDropped.emit(row)
            event.setDropAction(QtCore.Qt.DropAction.CopyAction)
            event.accept()
            return
        local_files = [url.toLocalFile() for url in mime_data.urls() if url.isLocalFile()]
        if local_files:
            self.fileDropped.emit(local_files[0])
            event.setDropAction(QtCore.Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dropEvent(event)


@dataclass(slots=True)
class FileEntry:
    path: Path
    loaded: LoadedFile | None = None
    error: str = ""
    enabled: bool = True
    curve_index: int | None = None
    display_name: str = ""


_EXPANDED_CONTAINER_SUFFIXES = {".csv", ".spc"}
_X_LABEL_OPTIONS = (
    "Raman shift / cm⁻¹",
    "2θ / °",
    "q / nm⁻¹",
    "q / Å⁻¹",
    "x",
)


def _curve_view(loaded: LoadedFile, curve_index: int) -> LoadedFile:
    """Create a lightweight one-curve view without copying the source arrays."""
    return LoadedFile(
        path=loaded.path,
        x=loaded.x,
        ys=loaded.ys[curve_index : curve_index + 1],
        header=loaded.header,
        labels=[loaded.labels[curve_index]],
        detected_delimiter=loaded.detected_delimiter,
        data_type=loaded.data_type,
        x_label=loaded.x_label,
    )


def _entries_from_loaded(path: Path, loaded: LoadedFile) -> list[FileEntry]:
    if path.suffix.lower() not in _EXPANDED_CONTAINER_SUFFIXES or loaded.curves <= 1:
        return [FileEntry(path=path, loaded=loaded, display_name=path.name)]
    return [
        FileEntry(
            path=path,
            loaded=_curve_view(loaded, curve_index),
            curve_index=curve_index,
            display_name=loaded.labels[curve_index],
        )
        for curve_index in range(loaded.curves)
    ]


def _load_reference_source(settings: AppSettings) -> LoadedFile:
    """Load the selected file or one persisted spectrum from a container file."""
    reference = load_file(settings.reference.filename, settings.import_options)
    curve_index = settings.reference.curve_index
    if curve_index is None:
        return reference
    if not 0 <= curve_index < reference.curves:
        raise DataValidationError(
            f"Reference spectrum {curve_index + 1} no longer exists in {reference.path.name}"
        )
    return _curve_view(reference, curve_index)


class ProcessingWorker(QtCore.QObject):
    progress = QtCore.Signal(str, int, int, str)
    finished = QtCore.Signal(object, object)
    failed = QtCore.Signal(str, str)
    cancelled = QtCore.Signal()

    def __init__(self, files: list[LoadedFile], settings: AppSettings, dataset_name: str):
        super().__init__()
        self.files = files
        self.settings = settings
        self.dataset_name = dataset_name
        self._cancel_event = threading.Event()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            dataset = combine_files(self.files, self.settings.import_options.alignment)
            reference = None
            if self.settings.reference.enabled:
                reference = _load_reference_source(self.settings)
            result = process_dataset(
                dataset,
                self.settings,
                reference_file=reference,
                progress=self.progress.emit,
                cancelled=self._cancel_event.is_set,
            )
            if self._cancel_event.is_set():
                self.cancelled.emit()
                return
            self.progress.emit("export", 0, 1, "Writing output files")
            output = Path(self.settings.output.directory)
            if not output.is_absolute():
                output = self.files[0].path.parent / output
            manifest = export_result(result, self.settings, output, self.dataset_name)
            self.progress.emit("export", 1, 1, f"Wrote {len(manifest.files)} files")
            self.finished.emit(result, manifest)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())

    def request_cancel(self) -> None:
        self._cancel_event.set()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings: AppSettings | None = None):
        super().__init__()
        self.settings = settings or AppSettings()
        self.entries: list[FileEntry] = []
        self.last_result: ProcessingResult | None = None
        self.last_manifest: ExportManifest | None = None
        self.worker: ProcessingWorker | None = None
        self.worker_thread: QtCore.QThread | None = None
        self._close_after_cancel = False
        self._elapsed_timer = QtCore.QElapsedTimer()
        self._last_output_directory = ""
        self._roi_pick_start: float | None = None
        self._reference_curve_index: int | None = None
        self._reference_curve_label = ""
        self._worker_count = 0
        self._detected_data_type = "generic"
        self._x_label_manually_selected = False
        self._adjusting_file_columns = False
        self._file_column_minimums = (46, 110, 58, 64, 64)
        self._file_data_min_width = self._file_column_minimums[1]
        self._live_preview_timer = QtCore.QTimer(self)
        self._live_preview_timer.setSingleShot(True)
        self._live_preview_timer.setInterval(160)
        self._live_preview_timer.timeout.connect(self._render_live_preview)

        self.setWindowTitle("Background Correction Studio")
        self.setMinimumSize(1180, 760)
        self.resize(1480, 900)
        self.setAcceptDrops(True)
        self._build_ui()
        self._create_menu()
        self._apply_settings(self.settings)
        self._connect_live_preview_controls()
        self._install_wheel_guard()
        self._set_style()
        self._log("Ready. Add spectral data files or drop them onto the window.")

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)
        splitter.addWidget(self._build_file_panel())
        splitter.addWidget(self._build_workspace())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([580, 900])

        run_bar = QtWidgets.QFrame()
        run_layout = QtWidgets.QHBoxLayout(run_bar)
        run_layout.setContentsMargins(0, 4, 0, 0)
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setMinimumWidth(260)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_processing)
        self.run_button = QtWidgets.QPushButton("Run correction")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self._start_processing)
        run_layout.addWidget(self.status_label)
        run_layout.addWidget(self.progress_bar, 1)
        run_layout.addWidget(self.cancel_button)
        run_layout.addWidget(self.run_button)
        main_layout.addWidget(run_bar)

    def _build_file_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)

        title_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Data series")
        title.setObjectName("panelTitle")
        self.dataset_name_edit = QtWidgets.QLineEdit("dataset")
        self.dataset_name_edit.setPlaceholderText("Dataset name")
        self.data_type_combo = self._combo([
            ("Automatic", "auto"),
            ("PXRD / XRD", "pxrd"),
            ("Raman", "raman"),
            ("Generic XY", "generic"),
        ])
        self.data_type_combo.setMinimumWidth(190)
        self.data_type_combo.setToolTip(
            "Automatically identify Raman or X-ray diffraction data from file metadata, headers, and axis values, "
            "or override the result manually."
        )
        self.data_type_combo.currentIndexChanged.connect(self._data_type_selection_changed)
        title_row.addWidget(title)
        title_row.addStretch()
        layout.addLayout(title_row)
        details_row = QtWidgets.QHBoxLayout()
        details_row.addWidget(QtWidgets.QLabel("Name"))
        details_row.addWidget(self.dataset_name_edit, 1)
        details_row.addWidget(QtWidgets.QLabel("Data type"))
        details_row.addWidget(self.data_type_combo)
        layout.addLayout(details_row)

        button_row = QtWidgets.QHBoxLayout()
        add_files = QtWidgets.QPushButton("Add files…")
        add_files.clicked.connect(self._choose_files)
        add_files.setToolTip("Select one or more spectral data files and add them to this series.")
        add_folder = QtWidgets.QPushButton("Add data folder…")
        add_folder.clicked.connect(self._choose_folder)
        add_folder.setToolTip("Import every supported data file found in the selected folder.")
        reload_button = QtWidgets.QPushButton("Reload")
        reload_button.clicked.connect(self._reload_entries)
        reload_button.setToolTip("Read all listed files again using the current Import settings.")
        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.clicked.connect(self._clear_files)
        clear_button.setToolTip("Remove every file from the list. Source files are never deleted.")
        for button in (add_files, add_folder, reload_button, clear_button):
            button_row.addWidget(button)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.file_table = ReorderTableWidget(0, 5)
        self.file_table.setHorizontalHeaderLabels(["Use", "Data item", "Points", "x min", "x max"])
        self.file_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.file_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.file_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.file_table.setDragEnabled(True)
        self.file_table.setAcceptDrops(True)
        self.file_table.viewport().setAcceptDrops(True)
        self.file_table.setDropIndicatorShown(True)
        self.file_table.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.file_table.setDragDropOverwriteMode(False)
        self.file_table.setDefaultDropAction(QtCore.Qt.DropAction.CopyAction)
        self.file_table.setToolTip("Drag a row up or down to change processing and plotting order.")
        self.file_table.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        self.file_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.file_table.viewport().installEventFilter(self)
        self.file_table.verticalHeader().setVisible(False)
        header = self.file_table.horizontalHeader()
        for column in (0, 2, 3):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setMinimumSectionSize(38)
        for column, width in enumerate((46, 250, 68, 78, 78)):
            header.resizeSection(column, width)
        header.sectionResized.connect(self._file_column_resized)
        self.file_table.itemSelectionChanged.connect(self._update_preview)
        self.file_table.itemChanged.connect(self._table_item_changed)
        self.file_table.rowsReordered.connect(self._reorder_entry)
        layout.addWidget(self.file_table, 1)

        order_row = QtWidgets.QHBoxLayout()
        up_button = QtWidgets.QPushButton("Move up")
        down_button = QtWidgets.QPushButton("Move down")
        remove_button = QtWidgets.QPushButton("Remove")
        up_button.setToolTip("Move the selected item one row earlier in the series.")
        down_button.setToolTip("Move the selected item one row later in the series.")
        remove_button.setToolTip("Remove the selected item from the list.")
        up_button.clicked.connect(lambda: self._move_entry(-1))
        down_button.clicked.connect(lambda: self._move_entry(1))
        remove_button.clicked.connect(self._remove_selected)
        order_row.addWidget(up_button)
        order_row.addWidget(down_button)
        order_row.addWidget(remove_button)
        order_row.addStretch()
        self.file_summary = QtWidgets.QLabel("0 items · 0 source files")
        order_row.addWidget(self.file_summary)
        layout.addLayout(order_row)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        self.log_view.setPlaceholderText("Activity and validation messages")
        self.log_view.setMaximumHeight(155)
        layout.addWidget(self.log_view)
        return panel

    def _build_workspace(self) -> QtWidgets.QWidget:
        workspace = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.preview_panel = self._build_preview_tab()
        splitter.addWidget(self.preview_panel)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._scroll_tab(self._build_processing_tab()), "Processing")
        self.tabs.addTab(self._scroll_tab(self._build_reference_tab()), "Reference")
        self.tabs.addTab(self._scroll_tab(self._build_plot_tab()), "Plot")
        self.tabs.addTab(self._build_roi_tab(), "ROIs")
        self.tabs.addTab(self._scroll_tab(self._build_output_tab()), "Output")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([520, 340])
        layout.addWidget(splitter)
        return workspace

    def _build_preview_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        toolbar_row = QtWidgets.QHBoxLayout()
        self.preview_mode = QtWidgets.QComboBox()
        self.preview_mode.addItems(["Imported curve", "Live correction", "Processed batch", "Result heatmap"])
        self.preview_mode.setCurrentIndex(1)
        self.preview_mode.currentIndexChanged.connect(self._update_preview)
        self.preview_mode.setToolTip("Choose the imported curve, live correction, processed curve, or result heatmap.")
        self.split_reference_check = QtWidgets.QCheckBox("Separate reference and baseline")
        self.split_reference_check.setToolTip("Show the scaled jar/reference signal and the additional fitted baseline separately in corrected curve previews.")
        self.split_reference_check.toggled.connect(self._update_preview)
        self.curve_spin = QtWidgets.QSpinBox()
        self.curve_spin.setRange(1, 1)
        self.curve_spin.setToolTip("Select a spectrum within the current data item.")
        self.curve_spin.valueChanged.connect(self._update_preview)
        self.preview_caption = QtWidgets.QLabel("No data selected")
        toolbar_row.addWidget(QtWidgets.QLabel("View"))
        toolbar_row.addWidget(self.preview_mode)
        self.curve_label = QtWidgets.QLabel("Curve")
        toolbar_row.addWidget(self.curve_label)
        toolbar_row.addWidget(self.curve_spin)
        toolbar_row.addWidget(self.split_reference_check)
        toolbar_row.addWidget(self.preview_caption, 1)
        layout.addLayout(toolbar_row)

        self.preview_figure = Figure(figsize=(7, 5), dpi=100)
        self.preview_canvas = FigureCanvasQTAgg(self.preview_figure)
        self.preview_canvas.mpl_connect("button_press_event", self._on_preview_click)
        self.preview_toolbar = NavigationToolbar2QT(self.preview_canvas, widget)
        layout.addWidget(self.preview_toolbar)
        layout.addWidget(self.preview_canvas, 1)
        self.curve_label.setVisible(False)
        self.curve_spin.setVisible(False)
        draw_curve_preview(self.preview_figure, [0, 1], [0, 0], title="Add data to begin")
        return widget

    def _build_processing_tab(self) -> QtWidgets.QWidget:
        widget, layout = self._form_container()

        import_group, form = self._group_form("Import")
        self.auto_header_check = QtWidgets.QCheckBox("Detect header rows automatically")
        self.header_rows_spin = QtWidgets.QSpinBox(); self.header_rows_spin.setRange(0, 100000)
        self.delimiter_combo = self._combo([("Auto", "auto"), ("Whitespace", "whitespace"), ("Tab", "tab"), ("Comma", "comma"), ("Semicolon", "semicolon")])
        self.alignment_combo = self._combo([("Require matching axes", "strict"), ("Interpolate to first axis", "interpolate")])
        self.recursive_check = QtWidgets.QCheckBox("Include subfolders")
        self.auto_header_check.setToolTip("Find the first numeric row automatically and treat earlier comment/text rows as the header.")
        self.header_rows_spin.setToolTip("Number of lines to skip when automatic header detection is disabled.")
        self.delimiter_combo.setToolTip("Column separator used in text files. Auto detects common separators from the first numeric row.")
        self.alignment_combo.setToolTip("Strict requires identical x values. Interpolation maps later files onto the first file's x-axis without extrapolation.")
        self.recursive_check.setToolTip("When adding a data folder, also search every folder below it.")
        form.addRow(self.auto_header_check)
        form.addRow("Header rows", self.header_rows_spin)
        form.addRow("Delimiter", self.delimiter_combo)
        form.addRow("x-axis alignment", self.alignment_combo)
        form.addRow(self.recursive_check)
        self._add_reset_row(form, "import")
        layout.addWidget(import_group)

        data_group, form = self._group_form("Data axis and range")
        self.data_form = form
        self.range_check = QtWidgets.QCheckBox("Restrict x-range")
        self.range_start_spin = self._float_spin(-1e12, 1e12, 6)
        self.range_stop_spin = self._float_spin(-1e12, 1e12, 6)
        self.range_start_spin.setSingleStep(0.1); self.range_stop_spin.setSingleStep(0.1)
        self.input_unit_combo = self._unit_combo()
        self.output_unit_combo = self._unit_combo()
        self.wavelength_spin = self._float_spin(1e-6, 1e6, 6); self.wavelength_spin.setSuffix(" Å"); self.wavelength_spin.setSingleStep(0.0001)
        self.range_check.setToolTip("Process only points between Start and Stop. Disable this to retain the complete imported axis.")
        self.range_start_spin.setToolTip("Lower boundary of the x-range retained for processing and export.")
        self.range_stop_spin.setToolTip("Upper boundary of the x-range retained for processing and export.")
        self.input_unit_combo.setToolTip("Physical unit currently represented by the imported x column.")
        self.output_unit_combo.setToolTip("Desired x unit. Choose Unchanged to perform no physical conversion.")
        self.wavelength_spin.setToolTip("X-ray wavelength used only when converting between scattering vector q and 2θ.")
        form.addRow(self.range_check)
        form.addRow("Start", self.range_start_spin)
        form.addRow("Stop", self.range_stop_spin)
        form.addRow("Input unit", self.input_unit_combo)
        form.addRow("Output unit", self.output_unit_combo)
        form.addRow("Wavelength", self.wavelength_spin)
        self.pxrd_only_controls = (
            self.input_unit_combo,
            self.output_unit_combo,
            self.wavelength_spin,
        )
        self._add_reset_row(form, "data")
        layout.addWidget(data_group)

        baseline_group, form = self._group_form("Background subtraction")
        self.baseline_check = QtWidgets.QCheckBox("Enable background subtraction")
        self.algorithm_combo = self._combo([("Asymmetric least squares (ALS)", "als"), ("Adaptive reweighted PLS (arPLS)", "arpls")])
        self.lambda_spin = self._float_spin(1e-9, 1e15, 6)
        self.asymmetry_spin = self._float_spin(1e-9, 0.999999, 8)
        self.iterations_spin = QtWidgets.QSpinBox(); self.iterations_spin.setRange(1, 100000)
        self.tolerance_spin = self._float_spin(0.0, 1.0, 10); self.tolerance_spin.setMinimumWidth(180)
        self.lambda_spin.setSingleStep(1.0)
        self.asymmetry_spin.setSingleStep(0.001)
        self.iterations_spin.setSingleStep(5)
        self.tolerance_spin.setSingleStep(0.0000001)
        self.baseline_check.setToolTip("Subtract a smooth estimated background from every curve.")
        self.algorithm_combo.setToolTip("ALS reproduces the legacy asymmetric fit. arPLS adapts its weights from negative residuals and needs less asymmetry tuning.")
        self.lambda_spin.setToolTip("Smoothness penalty. Larger values make the fitted background smoother and less able to follow narrow peaks.")
        self.asymmetry_spin.setToolTip("ALS weight for points above the baseline. Smaller values reject positive peaks more strongly; typical values are 0.001–0.1.")
        self.iterations_spin.setToolTip("Maximum number of reweighting/solve cycles allowed for each curve.")
        self.tolerance_spin.setToolTip("Stop early when the relative change falls below this value. Smaller values are stricter and may require more iterations.")
        form.addRow(self.baseline_check)
        form.addRow("Algorithm", self.algorithm_combo)
        form.addRow("Lambda (smoothness)", self.lambda_spin)
        form.addRow("Asymmetry", self.asymmetry_spin)
        form.addRow("Maximum iterations", self.iterations_spin)
        form.addRow("Convergence tolerance", self.tolerance_spin)
        self._add_reset_row(form, "baseline")
        layout.addWidget(baseline_group)

        norm_group, form = self._group_form("Post-processing")
        self.ground_check = QtWidgets.QCheckBox("Shift minimum to zero")
        self.normalization_combo = self._combo([("None", "none"), ("Area", "area"), ("Sum", "sum"), ("Absolute maximum", "max")])
        self.ground_check.setToolTip("After subtraction, shift each curve vertically so its minimum value becomes zero.")
        self.normalization_combo.setToolTip("Scale each final curve by its integrated area, sum, or largest absolute value. None preserves corrected intensity units.")
        form.addRow(self.ground_check)
        form.addRow("Normalization", self.normalization_combo)
        self._add_reset_row(form, "normalization")
        layout.addWidget(norm_group)
        layout.addStretch()
        return widget

    def _build_reference_tab(self) -> QtWidgets.QWidget:
        widget, layout = self._form_container()
        group, form = self._group_form("Jar / reference subtraction")
        self.reference_check = QtWidgets.QCheckBox("Enable reference subtraction")
        self.reference_path_edit = ReferenceDropLineEdit()
        self.reference_path_edit.setPlaceholderText("Browse or drop one data item here")
        self.reference_path_edit.itemDropped.connect(self._set_reference_from_entry)
        self.reference_path_edit.fileDropped.connect(self._set_reference_from_path)
        self.reference_path_edit.textEdited.connect(self._reference_path_edited)
        self.reference_use_selected_button = QtWidgets.QPushButton("Use selected item")
        self.reference_use_selected_button.clicked.connect(self._use_selected_as_reference)
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse_reference)
        browse.setToolTip("Choose a reference file from disk. To select a specific scan, drag its row from the data list.")
        path_row = QtWidgets.QWidget(); path_layout = QtWidgets.QHBoxLayout(path_row); path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.reference_path_edit, 1); path_layout.addWidget(self.reference_use_selected_button); path_layout.addWidget(browse)
        self.reference_source_label = QtWidgets.QLabel()
        self.reference_source_label.setWordWrap(True); self.reference_source_label.setObjectName("hintLabel")
        self.reference_range_check = QtWidgets.QCheckBox("Fit scaling factor inside a limited range")
        self.reference_start_spin = self._float_spin(-1e12, 1e12, 6)
        self.reference_stop_spin = self._float_spin(-1e12, 1e12, 6)
        self.reference_start_spin.setSingleStep(0.1); self.reference_stop_spin.setSingleStep(0.1)
        self.reference_check.setToolTip("Fit and subtract a scaled reference spectrum, such as an empty jar or sample holder, from every curve.")
        self.reference_path_edit.setToolTip(
            "Spectrum used as the subtraction reference. Drag one item from the data-series list to select an individual scan inside an SPC or CSV file."
        )
        self.reference_use_selected_button.setToolTip(
            "Use the currently selected data-series item as the reference and remove it from batch processing."
        )
        self.reference_range_check.setToolTip("Estimate the reference scale using only the selected x-range rather than the complete curve.")
        self.reference_start_spin.setToolTip("Lower x boundary used to estimate the reference scaling factor.")
        self.reference_stop_spin.setToolTip("Upper x boundary used to estimate the reference scaling factor.")
        form.addRow(self.reference_check)
        form.addRow("Reference source", path_row)
        form.addRow(self.reference_source_label)
        form.addRow(self.reference_range_check)
        form.addRow("Fit start", self.reference_start_spin)
        form.addRow("Fit stop", self.reference_stop_spin)
        self._add_reset_row(form, "reference")
        layout.addWidget(group)
        note = QtWidgets.QLabel(
            "Drop a data item above to use that exact spectrum, including one scan embedded in a Raman SPC file. "
            "The selected reference remains visible in the list but is unchecked from processing. The reference is "
            "interpolated onto the dataset axis; its corrected signal is fitted to each curve while the scaled raw "
            "reference is subtracted to preserve legacy behavior."
        )
        note.setWordWrap(True); note.setObjectName("hintLabel")
        layout.addWidget(note); layout.addStretch()
        return widget

    def _build_plot_tab(self) -> QtWidgets.QWidget:
        widget, layout = self._form_container()
        group, form = self._group_form("Result plot")
        self.plot_check = QtWidgets.QCheckBox("Generate result heatmap")
        self.colormap_combo = QtWidgets.QComboBox(); self.colormap_combo.addItems(["hot", "viridis", "magma", "inferno", "plasma", "cividis", "gray", "turbo", "coolwarm"])
        self.x_label_combo = QtWidgets.QComboBox()
        self.x_label_combo.setEditable(True)
        self.x_label_combo.addItems(list(_X_LABEL_OPTIONS))
        self.x_label_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.x_label_combo.activated.connect(self._x_label_selected)
        if self.x_label_combo.lineEdit() is not None:
            self.x_label_combo.lineEdit().editingFinished.connect(self._x_label_selected)
        self.automatic_time_axis_check = QtWidgets.QCheckBox("Automatic time axis")
        self.scan_interval_spin = self._float_spin(0.001, 1e9, 3)
        self.scan_interval_spin.setSuffix(" s")
        self.scan_interval_spin.setSingleStep(1.0)
        self.automatic_time_axis_check.toggled.connect(self.scan_interval_spin.setEnabled)
        self.automatic_time_axis_check.toggled.connect(lambda: self._update_preview())
        self.scan_interval_spin.editingFinished.connect(self._update_preview)
        self.colorbar_check = QtWidgets.QCheckBox("Show colorbar")
        self.flip_x_data_check = QtWidgets.QCheckBox("Flip x data")
        self.flip_y_data_check = QtWidgets.QCheckBox("Flip series order")
        self.flip_x_axis_check = QtWidgets.QCheckBox("Reverse x-axis")
        self.flip_y_axis_check = QtWidgets.QCheckBox("Reverse y-axis")
        self.plot_check.setToolTip("Create and export a two-dimensional heatmap of all corrected curves.")
        self.colormap_combo.setToolTip("Color palette used for the result heatmap.")
        self.x_label_combo.setToolTip(
            "Label displayed below the x-axis. Choose a prepared scientific label or type a custom one."
        )
        self.automatic_time_axis_check.setToolTip("Convert scan rows into elapsed minutes. Disable this to display scan numbers instead.")
        self.scan_interval_spin.setToolTip("Acquisition time represented by one scan. For example, 30 s produces heatmap rows spaced by 0.5 min.")
        self.colorbar_check.setToolTip("Add a color scale indicating the plotted intensity values.")
        self.flip_x_data_check.setToolTip("Reverse the data columns before plotting, without changing the numerical export.")
        self.flip_y_data_check.setToolTip("Reverse the displayed order of curves, without changing the numerical export.")
        self.flip_x_axis_check.setToolTip("Display the x-axis from high to low while retaining data order.")
        self.flip_y_axis_check.setToolTip("Display the series/time axis from high to low.")
        form.addRow(self.plot_check)
        form.addRow("Colormap", self.colormap_combo)
        form.addRow("x label", self.x_label_combo)
        form.addRow(self.automatic_time_axis_check)
        form.addRow("Time per scan", self.scan_interval_spin)
        form.addRow(self.colorbar_check)
        form.addRow(self.flip_x_data_check)
        form.addRow(self.flip_y_data_check)
        form.addRow(self.flip_x_axis_check)
        form.addRow(self.flip_y_axis_check)
        self._add_reset_row(form, "plot")
        layout.addWidget(group)

        limits_group, form = self._group_form("Color scale and image")
        self.display_scaling_combo = self._combo([
            ("Linear (global intensity)", "linear"),
            ("Robust percentile (recommended)", "robust"),
            ("Asinh / HDR", "asinh"),
            ("Normalize each scan", "per_scan"),
        ])
        self.percentile_low_spin = self._float_spin(0.0, 49.9, 1)
        self.percentile_high_spin = self._float_spin(50.1, 100.0, 1)
        self.percentile_low_spin.setSuffix(" %")
        self.percentile_high_spin.setSuffix(" %")
        self.percentile_low_spin.setSingleStep(0.5)
        self.percentile_high_spin.setSingleStep(0.5)
        self.value_limits_check = QtWidgets.QCheckBox("Use fixed color limits")
        self.value_min_spin = self._float_spin(-1e15, 1e15, 8)
        self.value_max_spin = self._float_spin(-1e15, 1e15, 8)
        self.value_min_spin.setSingleStep(0.1); self.value_max_spin.setSingleStep(0.1)
        self.width_spin = self._float_spin(1.0, 100.0, 2); self.width_spin.setSuffix(" in"); self.width_spin.setSingleStep(0.25)
        self.height_spin = self._float_spin(1.0, 100.0, 2); self.height_spin.setSuffix(" in"); self.height_spin.setSingleStep(0.25)
        self.dpi_spin = QtWidgets.QSpinBox(); self.dpi_spin.setRange(50, 2400); self.dpi_spin.setSingleStep(10)
        self.image_format_combo = self._combo([("PNG", "png"), ("PDF", "pdf"), ("SVG", "svg"), ("TIFF", "tiff")])
        self.display_scaling_combo.setToolTip(
            "Linear uses one uncompressed intensity scale. Robust ignores extreme values when choosing colors. "
            "Asinh/HDR compresses dominant peaks while retaining raw intensity labels. Normalize each scan "
            "maximizes within-scan contrast but removes absolute intensity differences between scans."
        )
        self.percentile_low_spin.setToolTip(
            "Values below this percentile use the lowest color in Robust, HDR, and per-scan modes."
        )
        self.percentile_high_spin.setToolTip(
            "Values above this percentile use the highest color in Robust, HDR, and per-scan modes. "
            "Lower this value to reveal weaker features."
        )
        self.value_limits_check.setToolTip("Use fixed intensity limits so multiple exported heatmaps share the same color scale.")
        self.value_min_spin.setToolTip("Intensity mapped to the bottom of the colormap when fixed limits are enabled.")
        self.value_max_spin.setToolTip("Intensity mapped to the top of the colormap when fixed limits are enabled.")
        self.width_spin.setToolTip("Physical width of the exported plot in inches.")
        self.height_spin.setToolTip("Physical height of the exported plot in inches.")
        self.dpi_spin.setToolTip("Raster resolution for PNG/TIFF output. Higher values create larger files.")
        self.image_format_combo.setToolTip("PNG/TIFF are raster images; PDF/SVG preserve vector text and axes.")
        self.display_scaling_combo.currentIndexChanged.connect(self._update_plot_scaling_controls)
        self.display_scaling_combo.currentIndexChanged.connect(self._update_preview)
        self.percentile_low_spin.valueChanged.connect(self._update_preview)
        self.percentile_high_spin.valueChanged.connect(self._update_preview)
        self.value_limits_check.toggled.connect(self._update_plot_scaling_controls)
        self.value_limits_check.toggled.connect(self._update_preview)
        self.value_min_spin.valueChanged.connect(self._update_preview)
        self.value_max_spin.valueChanged.connect(self._update_preview)
        self.colormap_combo.currentIndexChanged.connect(self._update_preview)
        self.colorbar_check.toggled.connect(self._update_preview)
        form.addRow("Display scaling", self.display_scaling_combo)
        form.addRow("Lower percentile", self.percentile_low_spin)
        form.addRow("Upper percentile", self.percentile_high_spin)
        scaling_note = QtWidgets.QLabel(
            "Display scaling changes only the heatmap colors. Corrected and exported numerical data remain unchanged."
        )
        scaling_note.setWordWrap(True); scaling_note.setObjectName("hintLabel")
        form.addRow(scaling_note)
        form.addRow(self.value_limits_check)
        form.addRow("Minimum", self.value_min_spin)
        form.addRow("Maximum", self.value_max_spin)
        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        form.addRow("DPI", self.dpi_spin)
        form.addRow("Format", self.image_format_combo)
        self._add_reset_row(form, "plot_image")
        layout.addWidget(limits_group); layout.addStretch()
        return widget

    def _build_roi_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(widget)
        options = QtWidgets.QHBoxLayout()
        self.roi_check = QtWidgets.QCheckBox("Enable ROI integration")
        self.roi_selection_combo = self._combo([("Integrated area", "area"), ("Maximum", "max")])
        self.roi_normalization_combo = self._combo([("Fractions per curve", "total"), ("Linear mean total", "linear"), ("Global maximum", "max"), ("None", "none")])
        self.roi_check.setToolTip("Calculate values inside defined x-ranges and include them in the preview/export. Disable this to skip all ROI work.")
        self.roi_selection_combo.setToolTip("Integrated area sums the signal across the range; Maximum records the highest point inside it.")
        self.roi_normalization_combo.setToolTip("Controls how ROI values are scaled across each curve or across the full result.")
        self.roi_check.toggled.connect(self._roi_enabled_changed)
        options.addWidget(self.roi_check); options.addStretch(); options.addWidget(QtWidgets.QLabel("Measure")); options.addWidget(self.roi_selection_combo)
        options.addWidget(QtWidgets.QLabel("Normalize")); options.addWidget(self.roi_normalization_combo)
        layout.addLayout(options)
        self.roi_table = QtWidgets.QTableWidget(0, 4)
        self.roi_table.setHorizontalHeaderLabels(["Name", "Start", "Stop", "Color"])
        self.roi_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.itemChanged.connect(self._roi_item_changed)
        layout.addWidget(self.roi_table, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.roi_pick_button = QtWidgets.QPushButton("Pick ROI on plot")
        self.roi_pick_button.setCheckable(True)
        self.roi_pick_button.setEnabled(False)
        self.roi_pick_button.setToolTip(
            "Click this button, then click the ROI start and stop positions in the preview above."
        )
        self.roi_pick_button.toggled.connect(self._toggle_roi_picker)
        add = QtWidgets.QPushButton("Add ROI"); add.clicked.connect(self._add_roi)
        remove = QtWidgets.QPushButton("Remove selected"); remove.clicked.connect(self._remove_roi)
        add.setToolTip("Add an editable ROI row manually. You can also use Pick ROI on plot in Preview.")
        remove.setToolTip("Remove the selected ROI rows.")
        reset = QtWidgets.QPushButton("Reset section"); reset.clicked.connect(lambda: self._reset_section("rois"))
        reset.setToolTip("Restore default ROI settings and remove all ROI ranges.")
        buttons.addWidget(self.roi_pick_button); buttons.addWidget(add); buttons.addWidget(remove); buttons.addStretch(); buttons.addWidget(reset); layout.addLayout(buttons)
        return widget

    def _build_output_tab(self) -> QtWidgets.QWidget:
        widget, layout = self._form_container()
        group, form = self._group_form("Output")
        self.output_delimiter_combo = self._combo([("Tab", "\t"), ("Comma", ","), ("Semicolon", ";")])
        self.save_corrected_check = QtWidgets.QCheckBox("Corrected/final data")
        self.save_baseline_check = QtWidgets.QCheckBox("Fitted baselines")
        self.save_original_check = QtWidgets.QCheckBox("Ranged original data")
        self.save_reference_check = QtWidgets.QCheckBox("Reference-corrected data")
        self.save_plot_check = QtWidgets.QCheckBox("Plot")
        self.save_rois_check = QtWidgets.QCheckBox("ROI table")
        destination = QtWidgets.QLabel("You will be asked for the destination folder when you click Run correction.")
        destination.setWordWrap(True); destination.setObjectName("hintLabel")
        form.addRow("Destination", destination)
        self.output_delimiter_combo.setToolTip("Column separator used in exported numerical data files.")
        self.save_corrected_check.setToolTip("Export the final processed curves after reference, baseline, grounding, and normalization steps.")
        self.save_baseline_check.setToolTip("Export the fitted background curve for every input curve.")
        self.save_original_check.setToolTip("Export imported intensity data after applying the selected x-range.")
        self.save_reference_check.setToolTip("Export curves after reference subtraction but before baseline subtraction.")
        self.save_plot_check.setToolTip("Save the configured heatmap image when result plotting is enabled.")
        self.save_rois_check.setToolTip("Save a CSV table containing calculated ROI values when ROI integration is enabled.")
        form.addRow("Data delimiter", self.output_delimiter_combo)
        form.addRow(self.save_corrected_check)
        form.addRow(self.save_baseline_check)
        form.addRow(self.save_original_check)
        form.addRow(self.save_reference_check)
        form.addRow(self.save_plot_check)
        form.addRow(self.save_rois_check)
        self._add_reset_row(form, "output")
        layout.addWidget(group)
        note = QtWidgets.QLabel("Every run includes a JSON manifest with all parameters and source filenames. Use the controls above to choose which additional data and plot files are written.")
        note.setWordWrap(True); note.setObjectName("hintLabel"); layout.addWidget(note); layout.addStretch()
        return widget

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        add_action = file_menu.addAction("Add files…"); add_action.setShortcut(QtGui.QKeySequence.StandardKey.Open); add_action.triggered.connect(self._choose_files)
        file_menu.addSeparator()
        load_project = file_menu.addAction("Open project…"); load_project.triggered.connect(self._load_project)
        save_project = file_menu.addAction("Save project…"); save_project.setShortcut(QtGui.QKeySequence.StandardKey.Save); save_project.triggered.connect(self._save_project)
        import_settings = file_menu.addAction("Import settings…"); import_settings.triggered.connect(self._import_settings)
        export_settings = file_menu.addAction("Export settings…"); export_settings.triggered.connect(self._export_settings)
        file_menu.addSeparator(); file_menu.addAction("Exit", self.close)
        self.settings_menu = self.menuBar().addMenu("Settings")
        workers_menu = self.settings_menu.addMenu("CPU workers")
        workers_menu.setToolTipsVisible(True)
        self.worker_action_group = QtGui.QActionGroup(self)
        self.worker_action_group.setExclusive(True)
        self.worker_actions: dict[int, QtGui.QAction] = {}
        choices = [("Auto (recommended)", 0)] + [
            (str(count), count) for count in range(1, max(2, os.cpu_count() or 1) + 1)
        ]
        for label, value in choices:
            action = workers_menu.addAction(label)
            action.setCheckable(True)
            action.setData(value)
            action.setToolTip(
                "Choose workers automatically from dataset size and available CPU threads."
                if value == 0 else f"Use exactly {value} CPU worker{'s' if value != 1 else ''}."
            )
            action.setStatusTip(action.toolTip())
            action.triggered.connect(
                lambda checked=False, selected=value: self._set_worker_count(selected) if checked else None
            )
            self.worker_action_group.addAction(action)
            self.worker_actions[value] = action
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("About", self._show_about)

    def _set_worker_count(self, workers: int) -> None:
        available = max(1, os.cpu_count() or 1)
        self._worker_count = 0 if workers <= 0 else min(int(workers), available)
        action = getattr(self, "worker_actions", {}).get(self._worker_count)
        if action is not None:
            action.setChecked(True)

    def _data_type_selection_changed(self, *_args) -> None:
        if not hasattr(self, "data_form"):
            return
        self.last_result = None
        self._update_data_type_ui()
        if self.entries:
            self._schedule_live_preview()

    def _x_label_selected(self, *_args) -> None:
        self._x_label_manually_selected = True
        if self.entries:
            self._update_preview()

    def _update_plot_scaling_controls(self, *_args) -> None:
        if not hasattr(self, "display_scaling_combo"):
            return
        mode = str(self.display_scaling_combo.currentData() or "robust")
        uses_percentiles = mode in {"robust", "asinh", "per_scan"}
        self.percentile_low_spin.setEnabled(uses_percentiles)
        self.percentile_high_spin.setEnabled(uses_percentiles)
        self.value_limits_check.setEnabled(mode == "linear")
        fixed_limits_active = mode == "linear" and self.value_limits_check.isChecked()
        self.value_min_spin.setEnabled(fixed_limits_active)
        self.value_max_spin.setEnabled(fixed_limits_active)

    def _refresh_detected_data_type(self) -> None:
        detected = {
            entry.loaded.data_type
            for entry in self.entries
            if entry.loaded is not None and entry.loaded.data_type in {"raman", "pxrd"}
        }
        if len(detected) == 1:
            self._detected_data_type = next(iter(detected))
        elif len(detected) > 1:
            self._detected_data_type = "mixed"
        else:
            self._detected_data_type = "generic"
        labels = {
            "raman": "Automatic (Raman detected)",
            "pxrd": "Automatic (PXRD/XRD detected)",
            "mixed": "Automatic (mixed data)",
            "generic": "Automatic (Generic XY)",
        }
        self.data_type_combo.setItemText(0, labels[self._detected_data_type])
        self._update_data_type_ui()

    def _effective_data_type(self) -> str:
        selected = str(self.data_type_combo.currentData() or "auto")
        if selected != "auto":
            return selected
        return "generic" if self._detected_data_type == "mixed" else self._detected_data_type

    def _suggested_x_label(self, data_type: str) -> str:
        labels = {
            entry.loaded.x_label
            for entry in self.entries
            if entry.loaded is not None
            and entry.loaded.data_type == data_type
            and entry.loaded.x_label
        }
        if len(labels) == 1:
            return next(iter(labels))
        if data_type == "raman":
            return "Raman shift / cm⁻¹"
        if data_type == "pxrd":
            return "q / nm⁻¹"
        return "x"

    def _update_data_type_ui(self) -> None:
        if not hasattr(self, "data_form"):
            return
        selected = str(self.data_type_combo.currentData() or "auto")
        effective = self._effective_data_type()
        show_pxrd_controls = effective == "pxrd"
        for control in self.pxrd_only_controls:
            label = self.data_form.labelForField(control)
            if label is not None:
                label.setVisible(show_pxrd_controls)
            control.setVisible(show_pxrd_controls)

        # Raman and generic axes are already in their physical coordinates.
        # Never allow a hidden q/2-theta conversion to alter them.
        must_preserve_axis = effective in {"raman", "generic"} and (selected != "auto" or bool(self.entries))
        if must_preserve_axis:
            input_blocker = QtCore.QSignalBlocker(self.input_unit_combo)
            output_blocker = QtCore.QSignalBlocker(self.output_unit_combo)
            self._set_combo(self.input_unit_combo, "unchanged")
            self._set_combo(self.output_unit_combo, "unchanged")
            del input_blocker, output_blocker

        if hasattr(self, "x_label_combo") and not self._x_label_manually_selected:
            if selected != "auto" or self.entries:
                self.x_label_combo.setCurrentText(self._suggested_x_label(effective))

    def current_settings(self) -> AppSettings:
        settings = AppSettings()
        settings.import_options.auto_header = self.auto_header_check.isChecked()
        settings.import_options.header_rows = self.header_rows_spin.value()
        settings.import_options.delimiter = self.delimiter_combo.currentData()
        settings.import_options.alignment = self.alignment_combo.currentData()
        settings.import_options.recursive_folders = self.recursive_check.isChecked()
        settings.data.data_type = self.data_type_combo.currentData()
        settings.data.range_enabled = self.range_check.isChecked()
        settings.data.range_start = self.range_start_spin.value(); settings.data.range_stop = self.range_stop_spin.value()
        settings.data.input_unit = self.input_unit_combo.currentData(); settings.data.output_unit = self.output_unit_combo.currentData()
        settings.data.wavelength_angstrom = self.wavelength_spin.value()
        settings.baseline.enabled = self.baseline_check.isChecked(); settings.baseline.algorithm = self.algorithm_combo.currentData()
        settings.baseline.lam = self.lambda_spin.value(); settings.baseline.asymmetry = self.asymmetry_spin.value()
        settings.baseline.max_iterations = self.iterations_spin.value(); settings.baseline.tolerance = self.tolerance_spin.value()
        settings.normalization.ground = self.ground_check.isChecked(); settings.normalization.mode = self.normalization_combo.currentData()
        settings.runtime.workers = self._worker_count
        settings.reference.enabled = self.reference_check.isChecked(); settings.reference.filename = self.reference_path_edit.text().strip()
        settings.reference.curve_index = self._reference_curve_index; settings.reference.curve_label = self._reference_curve_label
        settings.reference.fit_range_enabled = self.reference_range_check.isChecked()
        settings.reference.fit_range_start = self.reference_start_spin.value(); settings.reference.fit_range_stop = self.reference_stop_spin.value()
        settings.plot.enabled = self.plot_check.isChecked(); settings.plot.colormap = self.colormap_combo.currentText()
        settings.plot.display_scaling = self.display_scaling_combo.currentData(); settings.plot.percentile_low = self.percentile_low_spin.value(); settings.plot.percentile_high = self.percentile_high_spin.value()
        settings.plot.x_label = self.x_label_combo.currentText().strip() or "x"; settings.plot.automatic_time_axis = self.automatic_time_axis_check.isChecked(); settings.plot.scan_interval_seconds = self.scan_interval_spin.value()
        settings.plot.colorbar = self.colorbar_check.isChecked(); settings.plot.flip_x_data = self.flip_x_data_check.isChecked()
        settings.plot.flip_y_data = self.flip_y_data_check.isChecked(); settings.plot.flip_x_axis = self.flip_x_axis_check.isChecked(); settings.plot.flip_y_axis = self.flip_y_axis_check.isChecked()
        settings.plot.value_limits_enabled = self.value_limits_check.isChecked(); settings.plot.value_min = self.value_min_spin.value(); settings.plot.value_max = self.value_max_spin.value()
        settings.plot.width_inches = self.width_spin.value(); settings.plot.height_inches = self.height_spin.value(); settings.plot.dpi = self.dpi_spin.value(); settings.plot.image_format = self.image_format_combo.currentData()
        settings.rois.enabled = self.roi_check.isChecked(); settings.rois.selection = self.roi_selection_combo.currentData(); settings.rois.normalization = self.roi_normalization_combo.currentData()
        settings.rois.ranges = self._read_roi_rows()
        settings.output.directory = self._last_output_directory; settings.output.delimiter = self.output_delimiter_combo.currentData()
        settings.output.save_corrected = self.save_corrected_check.isChecked(); settings.output.save_baseline = self.save_baseline_check.isChecked(); settings.output.save_original = self.save_original_check.isChecked()
        settings.output.save_reference_corrected = self.save_reference_check.isChecked(); settings.output.save_plot = self.save_plot_check.isChecked(); settings.output.save_rois = self.save_rois_check.isChecked()
        return settings

    def _apply_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self.auto_header_check.setChecked(settings.import_options.auto_header); self.header_rows_spin.setValue(settings.import_options.header_rows)
        self._set_combo(self.delimiter_combo, settings.import_options.delimiter); self._set_combo(self.alignment_combo, settings.import_options.alignment); self.recursive_check.setChecked(settings.import_options.recursive_folders)
        self._set_combo(self.data_type_combo, settings.data.data_type)
        self.range_check.setChecked(settings.data.range_enabled); self.range_start_spin.setValue(settings.data.range_start); self.range_stop_spin.setValue(settings.data.range_stop)
        self._set_combo(self.input_unit_combo, settings.data.input_unit); self._set_combo(self.output_unit_combo, settings.data.output_unit); self.wavelength_spin.setValue(settings.data.wavelength_angstrom)
        self.baseline_check.setChecked(settings.baseline.enabled); self._set_combo(self.algorithm_combo, settings.baseline.algorithm); self.lambda_spin.setValue(settings.baseline.lam); self.asymmetry_spin.setValue(settings.baseline.asymmetry)
        self.iterations_spin.setValue(settings.baseline.max_iterations); self.tolerance_spin.setValue(settings.baseline.tolerance)
        self.ground_check.setChecked(settings.normalization.ground); self._set_combo(self.normalization_combo, settings.normalization.mode); self._set_worker_count(settings.runtime.workers)
        self._reference_curve_index = settings.reference.curve_index
        self._reference_curve_label = settings.reference.curve_label
        self.reference_check.setChecked(settings.reference.enabled); self.reference_path_edit.setText(settings.reference.filename); self.reference_range_check.setChecked(settings.reference.fit_range_enabled)
        self._update_reference_source_label()
        self.reference_start_spin.setValue(settings.reference.fit_range_start); self.reference_stop_spin.setValue(settings.reference.fit_range_stop)
        self.plot_check.setChecked(settings.plot.enabled); self.colormap_combo.setCurrentText(settings.plot.colormap); self.x_label_combo.setCurrentText(settings.plot.x_label); self.automatic_time_axis_check.setChecked(settings.plot.automatic_time_axis); self.scan_interval_spin.setValue(settings.plot.scan_interval_seconds); self.scan_interval_spin.setEnabled(settings.plot.automatic_time_axis)
        self.colorbar_check.setChecked(settings.plot.colorbar); self.flip_x_data_check.setChecked(settings.plot.flip_x_data); self.flip_y_data_check.setChecked(settings.plot.flip_y_data); self.flip_x_axis_check.setChecked(settings.plot.flip_x_axis); self.flip_y_axis_check.setChecked(settings.plot.flip_y_axis)
        self._set_combo(self.display_scaling_combo, settings.plot.display_scaling); self.percentile_low_spin.setValue(settings.plot.percentile_low); self.percentile_high_spin.setValue(settings.plot.percentile_high)
        self.value_limits_check.setChecked(settings.plot.value_limits_enabled); self.value_min_spin.setValue(settings.plot.value_min); self.value_max_spin.setValue(settings.plot.value_max)
        self.width_spin.setValue(settings.plot.width_inches); self.height_spin.setValue(settings.plot.height_inches); self.dpi_spin.setValue(settings.plot.dpi); self._set_combo(self.image_format_combo, settings.plot.image_format)
        self._update_plot_scaling_controls()
        self.roi_check.setChecked(settings.rois.enabled); self._set_combo(self.roi_selection_combo, settings.rois.selection); self._set_combo(self.roi_normalization_combo, settings.rois.normalization); self._write_roi_rows(settings.rois.ranges)
        output_path = Path(settings.output.directory) if settings.output.directory else None
        self._last_output_directory = str(output_path) if output_path and output_path.is_absolute() else ""
        self._set_combo(self.output_delimiter_combo, settings.output.delimiter)
        self.save_corrected_check.setChecked(settings.output.save_corrected); self.save_baseline_check.setChecked(settings.output.save_baseline); self.save_original_check.setChecked(settings.output.save_original)
        self.save_reference_check.setChecked(settings.output.save_reference_corrected); self.save_plot_check.setChecked(settings.output.save_plot); self.save_rois_check.setChecked(settings.output.save_rois)
        self._x_label_manually_selected = False
        self._update_data_type_ui()

    def _choose_files(self) -> None:
        filters = "Spectral data (*.xy *.dat *.chi *.txt *.raman *.csv *.tsv *.spc);;All files (*)"
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add spectral data", "", filters)
        self._add_paths([Path(path) for path in files])

    def _choose_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Add data folder")
        if folder:
            paths = self._discover_files_with_feedback(Path(folder))
            self._add_paths(paths)

    def _discover_files_with_feedback(self, folder: Path) -> list[Path]:
        self.progress_bar.setRange(0, 0)
        self.status_label.setText(f"Scanning {folder.name}…")
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )
        try:
            paths = discover_files(folder, self.recursive_check.isChecked())
        finally:
            self.progress_bar.setRange(0, 100)
        if not paths:
            self.progress_bar.setValue(0)
            self.status_label.setText("No supported data files found")
        return paths

    def _update_file_progress(
        self,
        action: str,
        completed: int,
        total: int,
        filename: str,
        timer: QtCore.QElapsedTimer,
    ) -> None:
        percent = int(round(100 * completed / total)) if total else 0
        status = f"{action} {filename} · {completed}/{total}"
        if completed and completed < total and timer.isValid():
            elapsed_seconds = timer.elapsed() / 1000.0
            remaining_seconds = max(0.0, elapsed_seconds * (total - completed) / completed)
            status += f" · ETA {remaining_seconds:.1f} s"
        self.progress_bar.setValue(percent)
        self.status_label.setText(status)
        # File parsing remains in the foreground for minimum overhead, while
        # paint/timer events keep the progress display visibly moving.
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )

    def _add_paths(self, paths: list[Path]) -> None:
        if not paths:
            return
        known = {entry.path.resolve() for entry in self.entries}
        seen = set(known)
        new_paths: list[Path] = []
        for candidate in paths:
            path = Path(candidate)
            resolved = path.resolve()
            if path.suffix.lower() not in SUPPORTED_SUFFIXES or resolved in seen:
                continue
            seen.add(resolved)
            new_paths.append(path)
        if not new_paths:
            self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
            self.status_label.setText("No new supported data files found")
            return
        import_settings = self.current_settings().import_options
        self._live_preview_timer.stop()
        timer = QtCore.QElapsedTimer(); timer.start()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.status_label.setText(f"Importing 0/{len(new_paths)} files…")
        had_entries = bool(self.entries)
        loaded_count = 0
        for index, path in enumerate(new_paths, start=1):
            try:
                loaded = load_file(path, import_settings)
                added_entries = _entries_from_loaded(path, loaded)
                self.entries.extend(added_entries)
                loaded_count += 1
                item_note = f" as {len(added_entries)} data items" if len(added_entries) > 1 else ""
                self._log(f"Loaded {path.name}: {loaded.curves} curve(s), {loaded.points} points{item_note}")
            except Exception as exc:
                self.entries.append(FileEntry(path=path, error=str(exc), enabled=False, display_name=path.name))
                self._log(f"ERROR {path.name}: {exc}")
            self._update_file_progress("Importing", index, len(new_paths), path.name, timer)
        if not had_entries and self.entries:
            self.dataset_name_edit.setText(new_paths[0].stem)
        self.last_result = None
        self._refresh_table()
        self._autosize_file_columns()
        QtCore.QTimer.singleShot(0, self._autosize_file_columns)
        failed_count = len(new_paths) - loaded_count
        suffix = f" · {failed_count} failed" if failed_count else ""
        self.status_label.setText(
            f"Import complete · {loaded_count}/{len(new_paths)} files · {timer.elapsed() / 1000.0:.2f} s{suffix}"
        )

    def _reload_entries(self) -> None:
        if not self.entries:
            return
        import_settings = self.current_settings().import_options
        self._live_preview_timer.stop()
        timer = QtCore.QElapsedTimer(); timer.start()
        source_paths: list[Path] = []
        seen: set[Path] = set()
        for entry in self.entries:
            resolved = entry.path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                source_paths.append(entry.path)
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.status_label.setText(f"Reloading 0/{len(source_paths)} files…")
        loaded_sources: dict[Path, LoadedFile | Exception] = {}
        loaded_count = 0
        for index, path in enumerate(source_paths, start=1):
            try:
                loaded_sources[path.resolve()] = load_file(path, import_settings)
                loaded_count += 1
            except Exception as exc:
                loaded_sources[path.resolve()] = exc
            self._update_file_progress("Reloading", index, len(source_paths), path.name, timer)
        for entry in self.entries:
            source = loaded_sources[entry.path.resolve()]
            if isinstance(source, Exception):
                entry.loaded = None; entry.error = str(source); entry.enabled = False
                continue
            was_unavailable = entry.loaded is None
            try:
                if entry.curve_index is None:
                    entry.loaded = source
                else:
                    if entry.curve_index >= source.curves:
                        raise DataValidationError(
                            f"{entry.path.name}: curve {entry.curve_index + 1} no longer exists after reload"
                        )
                    entry.loaded = _curve_view(source, entry.curve_index)
                    entry.display_name = source.labels[entry.curve_index]
                entry.error = ""
                if was_unavailable:
                    entry.enabled = True
            except Exception as exc:
                entry.loaded = None; entry.error = str(exc); entry.enabled = False
        self.last_result = None; self._refresh_table(); self._autosize_file_columns(); QtCore.QTimer.singleShot(0, self._autosize_file_columns); self._log("Reloaded input files using the current import settings.")
        failed_count = len(source_paths) - loaded_count
        suffix = f" · {failed_count} failed" if failed_count else ""
        self.status_label.setText(
            f"Reload complete · {loaded_count}/{len(source_paths)} files · {timer.elapsed() / 1000.0:.2f} s{suffix}"
        )

    @QtCore.Slot(int, int, int)
    def _file_column_resized(self, logical_index: int, _old_size: int, _new_size: int) -> None:
        if self._adjusting_file_columns or logical_index in {1, 4}:
            return
        self._fit_file_columns_to_viewport(logical_index)

    def _autosize_file_columns(self) -> None:
        """Fit imported content with padding while keeping the table flush to its viewport."""
        if not hasattr(self, "file_table") or self.file_table.columnCount() != 5:
            return
        available = self.file_table.viewport().width()
        minimum_total = sum(self._file_column_minimums)
        if available < minimum_total:
            return
        margin = 14
        hints = [
            max(self._file_column_minimums[column], self.file_table.sizeHintForColumn(column) + margin)
            for column in range(5)
        ]
        # Keep metadata columns compact and let Data item absorb every remaining
        # pixel through its Stretch resize mode.
        fixed_columns = (0, 2, 3, 4)
        widths = {column: hints[column] for column in fixed_columns}
        budget = available - self._file_data_min_width
        excess = max(0, sum(widths.values()) - budget)
        for column in (3, 2, 0, 4):
            if excess <= 0:
                break
            reducible = max(0, widths[column] - self._file_column_minimums[column])
            reduction = min(excess, reducible)
            widths[column] -= reduction
            excess -= reduction
        self._adjusting_file_columns = True
        try:
            header = self.file_table.horizontalHeader()
            for column in fixed_columns:
                header.resizeSection(column, widths[column])
        finally:
            self._adjusting_file_columns = False
        self._fit_file_columns_to_viewport()

    def _fit_file_columns_to_viewport(self, changed_column: int | None = None) -> None:
        """Clamp interactive sections so the table can never require horizontal scrolling."""
        if self._adjusting_file_columns or not hasattr(self, "file_table"):
            return
        available = self.file_table.viewport().width()
        if available <= 0:
            return
        fixed_columns = (0, 2, 3, 4)
        budget = available - self._file_data_min_width
        header = self.file_table.horizontalHeader()
        excess = max(0, sum(header.sectionSize(column) for column in fixed_columns) - budget)
        if excess <= 0:
            return
        order = []
        if changed_column in fixed_columns:
            order.append(changed_column)
        order.extend(column for column in (3, 2, 0, 4) if column not in order)
        self._adjusting_file_columns = True
        try:
            for column in order:
                if excess <= 0:
                    break
                minimum = self._file_column_minimums[column]
                current = header.sectionSize(column)
                reduction = min(excess, max(0, current - minimum))
                if reduction:
                    header.resizeSection(column, current - reduction)
                    excess -= reduction
        finally:
            self._adjusting_file_columns = False

    def _refresh_table(self) -> None:
        self.file_table.blockSignals(True); self.file_table.setRowCount(len(self.entries))
        valid_curves = 0
        for row, entry in enumerate(self.entries):
            use = QtWidgets.QTableWidgetItem(); use.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsSelectable); use.setCheckState(QtCore.Qt.CheckState.Checked if entry.enabled else QtCore.Qt.CheckState.Unchecked)
            name_item = QtWidgets.QTableWidgetItem(entry.display_name or entry.path.name)
            name_item.setToolTip(f"Source: {entry.path}")
            self.file_table.setItem(row, 0, use); self.file_table.setItem(row, 1, name_item)
            if entry.loaded:
                loaded = entry.loaded; valid_curves += loaded.curves if entry.enabled else 0
                values = [str(loaded.points), f"{loaded.x_min:.6g}", f"{loaded.x_max:.6g}"]
            else:
                values = ["—", "—", "—"]
            for column, value in enumerate(values, start=2): self.file_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
            if entry.error:
                for column in range(5):
                    self.file_table.item(row, column).setForeground(QtGui.QColor("#d9534f"))
                    self.file_table.item(row, column).setToolTip(entry.error)
        source_count = len({entry.path.resolve() for entry in self.entries})
        self.file_table.blockSignals(False); self.file_summary.setText(
            f"{len(self.entries)} items · {source_count} source file{'s' if source_count != 1 else ''}"
        )
        if self.entries and self.file_table.currentRow() < 0: self.file_table.selectRow(0)
        self._refresh_detected_data_type()
        self._update_preview()

    def _table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() == 0 and 0 <= item.row() < len(self.entries):
            self.entries[item.row()].enabled = item.checkState() == QtCore.Qt.CheckState.Checked
            self._refresh_table()

    def _move_entry(self, offset: int) -> None:
        row = self.file_table.currentRow(); target = row + offset
        if row < 0 or target < 0 or target >= len(self.entries): return
        self._reorder_entry(row, target)

    @QtCore.Slot(int, int)
    def _reorder_entry(self, source: int, target: int) -> None:
        if source == target or not (0 <= source < len(self.entries)) or not (0 <= target < len(self.entries)):
            return
        entry = self.entries.pop(source)
        self.entries.insert(target, entry)
        self.last_result = None
        self._refresh_table()
        self.file_table.selectRow(target)
        self._log(f"Moved {entry.path.name} to position {target + 1}.")

    def _remove_selected(self) -> None:
        row = self.file_table.currentRow()
        if row >= 0: self.entries.pop(row); self.last_result = None; self._refresh_table()

    def _clear_files(self) -> None:
        self.entries.clear(); self.last_result = None; self.last_manifest = None; self._refresh_table(); self.preview_figure.clear(); self.preview_canvas.draw_idle()

    def _update_preview(self) -> None:
        mode = self.preview_mode.currentIndex()
        try:
            preview_settings = self.current_settings()
        except ValueError:
            preview_settings = self.settings
        if mode in {2, 3} and self.last_result is None:
            mode = 1
            self.preview_mode.blockSignals(True)
            self.preview_mode.setCurrentIndex(mode)
            self.preview_mode.blockSignals(False)
        if mode == 3 and self.last_result is not None:
            figure = create_result_figure(self.last_result, preview_settings.plot, preview_settings.rois)
            self.preview_figure.clear()
            # Move artists by recreating through the same plotting function on our canvas.
            result_figure = figure
            self.preview_canvas.figure = result_figure; self.preview_figure = result_figure; result_figure.set_canvas(self.preview_canvas)
            self.preview_caption.setText(f"{self.last_result.curve_count} curves × {self.last_result.point_count} points")
            self._draw_roi_pick_marker()
            self.preview_canvas.draw_idle(); return
        row = self.file_table.currentRow()
        if row < 0 or row >= len(self.entries) or self.entries[row].loaded is None:
            self.curve_label.setVisible(False); self.curve_spin.setVisible(False)
            return
        loaded = self.entries[row].loaded
        assert loaded is not None
        show_curve_selector = loaded.curves > 1
        self.curve_label.setVisible(show_curve_selector); self.curve_spin.setVisible(show_curve_selector)
        self.curve_spin.blockSignals(True); self.curve_spin.setRange(1, loaded.curves); self.curve_spin.setValue(min(self.curve_spin.value(), loaded.curves)); self.curve_spin.blockSignals(False)
        local_index = self.curve_spin.value() - 1
        if mode == 2 and self.last_result is not None:
            offset = sum(entry.loaded.curves for entry in self.entries[:row] if entry.enabled and entry.loaded)
            if self.entries[row].enabled and offset + local_index < self.last_result.curve_count:
                index = offset + local_index
                self._draw_processed_curve(self.last_result, index, preview_settings)
                self.preview_caption.setText(
                    f"Processed batch · {self.last_result.labels[index]} · {self.last_result.elapsed_seconds:.2f} s total"
                )
            else: mode = 0
        if mode == 1:
            try:
                preview_settings.runtime.workers = 1
                reference = None
                if preview_settings.reference.enabled:
                    if not preview_settings.reference.filename:
                        raise DataValidationError("Select a reference file to update the live correction")
                    reference = _load_reference_source(preview_settings)
                dataset = DatasetData(
                    x=loaded.x,
                    ys=loaded.ys[local_index : local_index + 1],
                    labels=[loaded.labels[local_index]],
                    source_paths=[loaded.path],
                )
                result = process_dataset(dataset, preview_settings, reference_file=reference)
                self._draw_processed_curve(result, 0, preview_settings, "Live correction · ")
                self.preview_caption.setText(
                    f"Live · λ {preview_settings.baseline.lam:g} · asymmetry {preview_settings.baseline.asymmetry:g} · {result.elapsed_seconds:.2f} s"
                )
            except Exception as exc:
                draw_curve_preview(
                    self.preview_figure,
                    loaded.x,
                    loaded.ys[local_index],
                    title="Live correction unavailable",
                    x_label=preview_settings.plot.x_label,
                    roi_settings=preview_settings.rois,
                )
                self.preview_caption.setText(str(exc))
        if mode == 0:
            draw_curve_preview(self.preview_figure, loaded.x, loaded.ys[local_index], title=loaded.labels[local_index], x_label=preview_settings.plot.x_label, roi_settings=preview_settings.rois)
            self.preview_caption.setText(f"{loaded.path.name} · curve {local_index + 1}/{loaded.curves}")
        self._draw_roi_pick_marker()
        self.preview_canvas.draw_idle()

    def _draw_processed_curve(self, result: ProcessingResult, index: int, settings: AppSettings, prefix: str = "") -> None:
        has_reference = np.isfinite(result.reference_scales[index])
        split = has_reference and self.split_reference_check.isChecked()
        baseline = result.baselines[index] if split else result.original[index] - result.corrected[index]
        label = "Additional fitted baseline" if split else "Total subtracted (JAR/reference + baseline)" if has_reference else "Baseline"
        draw_curve_preview(
            self.preview_figure, result.x, result.original[index], baseline,
            result.corrected[index], result.final[index], prefix + result.labels[index],
            settings.plot.x_label, settings.rois, label,
            reference=result.reference_scaled[index] if split else None,
        )

    def _connect_live_preview_controls(self) -> None:
        for control in (
            self.baseline_check,
            self.range_check,
            self.ground_check,
            self.reference_check,
            self.reference_range_check,
        ):
            control.toggled.connect(self._schedule_live_preview)
        for control in (
            self.algorithm_combo,
            self.input_unit_combo,
            self.output_unit_combo,
            self.normalization_combo,
        ):
            control.currentIndexChanged.connect(self._schedule_live_preview)
        for control in (
            self.lambda_spin,
            self.asymmetry_spin,
            self.iterations_spin,
            self.tolerance_spin,
            self.range_start_spin,
            self.range_stop_spin,
            self.wavelength_spin,
            self.reference_start_spin,
            self.reference_stop_spin,
        ):
            control.valueChanged.connect(self._schedule_live_preview)
        self.reference_path_edit.editingFinished.connect(self._schedule_live_preview)

    def _schedule_live_preview(self, *_args) -> None:
        if not self.entries or self.worker is not None:
            return
        self.preview_mode.blockSignals(True)
        self.preview_mode.setCurrentIndex(1)
        self.preview_mode.blockSignals(False)
        self._live_preview_timer.start()

    def _render_live_preview(self) -> None:
        if self.preview_mode.currentIndex() == 1 and self.worker is None:
            self._update_preview()

    def _start_processing(self) -> None:
        files = [entry.loaded for entry in self.entries if entry.enabled and entry.loaded is not None]
        if not files:
            self._show_error("No valid enabled input files are available."); return
        self._live_preview_timer.stop()
        try:
            settings = self.current_settings()
        except ValueError as exc:
            self._show_error(str(exc)); return
        if settings.reference.enabled and not settings.reference.filename:
            self._show_error("Reference correction is enabled, but no reference file is selected."); return
        if settings.plot.value_limits_enabled and settings.plot.value_min >= settings.plot.value_max:
            self._show_error("The fixed plot minimum must be lower than the maximum."); return
        initial_directory = self._last_output_directory
        if not initial_directory or not Path(initial_directory).is_dir():
            initial_directory = str(files[0].path.parent)
        output_directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose destination for this run",
            initial_directory,
            QtWidgets.QFileDialog.Option.ShowDirsOnly,
        )
        if not output_directory:
            self.status_label.setText("Ready")
            self._log("Run cancelled: no output destination was selected.")
            return
        self._last_output_directory = output_directory
        settings.output.directory = output_directory

        self.settings = settings; self.last_result = None; self.last_manifest = None
        self.worker_thread = QtCore.QThread(self)
        self.worker = ProcessingWorker(files, settings, self.dataset_name_edit.text().strip() or files[0].path.stem)
        self.worker.moveToThread(self.worker_thread); self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress); self.worker.finished.connect(self._on_finished); self.worker.failed.connect(self._on_failed); self.worker.cancelled.connect(self._on_cancelled)
        self.worker.finished.connect(self.worker_thread.quit); self.worker.failed.connect(self.worker_thread.quit); self.worker.cancelled.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._thread_finished)
        self._set_busy(True); self.progress_bar.setValue(0); self._elapsed_timer.start(); self._log("Starting processing…"); self.worker_thread.start()

    def _cancel_processing(self) -> None:
        if self.worker: self.worker.request_cancel(); self.status_label.setText("Cancelling…"); self.cancel_button.setEnabled(False); self._log("Cancellation requested.")

    @QtCore.Slot(str, int, int, str)
    def _on_progress(self, stage: str, completed: int, total: int, message: str) -> None:
        percent = int(round(100 * completed / total)) if total else 0
        status = message
        if stage == "process" and completed and total and self._elapsed_timer.isValid():
            elapsed_seconds = self._elapsed_timer.elapsed() / 1000.0
            remaining_seconds = max(0.0, elapsed_seconds * (total - completed) / completed)
            status = f"{message} · {completed}/{total} · ETA {remaining_seconds:.1f} s"
        self.progress_bar.setValue(percent); self.status_label.setText(status)
        if stage in {"prepare", "complete", "export"}: self._log(message)

    @QtCore.Slot(object, object)
    def _on_finished(self, result: ProcessingResult, manifest: ExportManifest) -> None:
        self.last_result = result; self.last_manifest = manifest; self.progress_bar.setValue(100)
        self.status_label.setText(f"Complete · {result.elapsed_seconds:.2f} s · {result.workers_used} worker(s)")
        self._log(f"Complete: {result.curve_count} curves, {result.point_count} points, {result.elapsed_seconds:.2f} s")
        self._log(f"Output: {manifest.output_directory}")
        self.preview_mode.setCurrentIndex(2); self._update_preview()

    @QtCore.Slot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        self.status_label.setText("Failed"); self._log(details); self._show_error(message)

    @QtCore.Slot()
    def _on_cancelled(self) -> None:
        self.status_label.setText("Cancelled"); self._log("Processing cancelled.")

    @QtCore.Slot()
    def _thread_finished(self) -> None:
        self._set_busy(False)
        if self.worker: self.worker.deleteLater()
        if self.worker_thread: self.worker_thread.deleteLater()
        self.worker = None; self.worker_thread = None
        if self._close_after_cancel:
            self._close_after_cancel = False
            QtCore.QTimer.singleShot(0, self.close)

    def _set_busy(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy); self.cancel_button.setEnabled(busy); self.tabs.setEnabled(not busy); self.preview_panel.setEnabled(not busy); self.file_table.setEnabled(not busy); self.settings_menu.setEnabled(not busy)

    def _save_project(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save project", f"{self.dataset_name_edit.text()}.bcproject", "Background Correction Project (*.bcproject)")
        if not filename: return
        source_files = list(dict.fromkeys(str(entry.path) for entry in self.entries))
        payload = {
            "project_version": 2,
            "dataset_name": self.dataset_name_edit.text(),
            "files": source_files,
            "items": [
                {
                    "path": str(entry.path),
                    "curve_index": entry.curve_index,
                    "enabled": entry.enabled,
                }
                for entry in self.entries
            ],
            "settings": self.current_settings().to_dict(),
        }
        Path(filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"); self._log(f"Saved project: {filename}")

    def _load_project(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open project", "", "Background Correction Project (*.bcproject)")
        if not filename: return
        try:
            payload = json.loads(Path(filename).read_text(encoding="utf-8")); settings = AppSettings.from_dict(payload["settings"]); self._apply_settings(settings)
            saved_items = payload.get("items", [])
            source_files = payload.get("files", [])
            if saved_items:
                source_files = list(dict.fromkeys(str(item["path"]) for item in saved_items))
            self.entries.clear(); self._add_paths([Path(path) for path in source_files]); self.dataset_name_edit.setText(payload.get("dataset_name", "dataset"))
            if saved_items:
                available = {
                    (str(entry.path.resolve()).casefold(), entry.curve_index): entry
                    for entry in self.entries
                }
                restored: list[FileEntry] = []
                used: set[int] = set()
                for item in saved_items:
                    key = (str(Path(item["path"]).resolve()).casefold(), item.get("curve_index"))
                    entry = available.get(key)
                    if entry is not None:
                        entry.enabled = bool(item.get("enabled", True))
                        restored.append(entry); used.add(id(entry))
                restored.extend(entry for entry in self.entries if id(entry) not in used)
                self.entries = restored
            else:
                for entry, enabled in zip(self.entries, payload.get("enabled", [])): entry.enabled = bool(enabled)
            self._refresh_table(); self._log(f"Opened project: {filename}")
        except Exception as exc: self._show_error(f"Could not open project: {exc}")

    def _import_settings(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import settings", "", "JSON settings (*.json);;All files (*)")
        if filename:
            try: self._apply_settings(AppSettings.load(filename)); self._log(f"Imported settings: {filename}")
            except Exception as exc: self._show_error(f"Could not import settings: {exc}")

    def _export_settings(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export settings", "background-correction-settings.json", "JSON settings (*.json)")
        if filename: self.current_settings().save(filename); self._log(f"Exported settings: {filename}")

    def _browse_reference(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select reference spectrum", "", "Spectral data (*.xy *.dat *.chi *.txt *.raman *.csv *.tsv *.spc)")
        if filename:
            self._set_reference_from_path(filename)

    def _set_reference_from_path(self, filename: str) -> None:
        path = Path(filename)
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            self._show_error("Drop a supported spectral data file onto the reference field.")
            return
        self._reference_curve_index = None
        self._reference_curve_label = ""
        self.reference_path_edit.setText(str(path))
        self._update_reference_source_label()
        self.reference_check.setChecked(True)
        self._schedule_live_preview()

    def _reference_path_edited(self, _text: str) -> None:
        # A manually entered path cannot retain an index from a previously
        # selected spectrum inside an SPC/CSV container.
        self._reference_curve_index = None
        self._reference_curve_label = ""
        self._update_reference_source_label()

    def _use_selected_as_reference(self) -> None:
        row = self.file_table.currentRow()
        if row < 0:
            self._show_error("Select a data item in the left list first.")
            return
        self._set_reference_from_entry(row)

    def _set_reference_from_entry(self, row: int) -> None:
        if not 0 <= row < len(self.entries):
            return
        entry = self.entries[row]
        if entry.loaded is None:
            self._show_error("The selected data item could not be loaded and cannot be used as a reference.")
            return

        curve_index = entry.curve_index
        curve_label = entry.display_name or entry.loaded.labels[0]
        if curve_index is None and entry.loaded.curves > 1:
            curve_index = min(max(self.curve_spin.value() - 1, 0), entry.loaded.curves - 1)
            curve_label = entry.loaded.labels[curve_index]

        self._reference_curve_index = curve_index
        self._reference_curve_label = curve_label if curve_index is not None else ""
        self.reference_path_edit.setText(str(entry.path))
        self._update_reference_source_label()
        self.reference_check.setChecked(True)

        # A background/reference scan normally should not also appear in the
        # corrected batch. Keep it listed for preview and make this reversible.
        entry.enabled = False
        self.last_result = None
        self._refresh_table()
        self.file_table.selectRow(row)
        selection = f"spectrum {curve_index + 1} ({curve_label})" if curve_index is not None else entry.path.name
        self._log(f"Reference set to {selection}; the source item was unchecked from processing.")
        self._schedule_live_preview()

    def _update_reference_source_label(self) -> None:
        if not hasattr(self, "reference_source_label"):
            return
        filename = self.reference_path_edit.text().strip()
        if not filename:
            self.reference_source_label.clear()
            self.reference_source_label.setVisible(False)
            return
        if self._reference_curve_index is None:
            text = "Using the first spectrum in the selected file."
        else:
            label = self._reference_curve_label or f"Spectrum {self._reference_curve_index + 1}"
            text = f"Using embedded spectrum {self._reference_curve_index + 1}: {label}"
        self.reference_source_label.setText(text)
        self.reference_source_label.setVisible(True)

    def _add_roi(self) -> None:
        self._append_roi(0.0, 1.0)

    def _append_roi(self, start: float, stop: float) -> None:
        self.roi_table.blockSignals(True)
        row = self.roi_table.rowCount(); self.roi_table.insertRow(row)
        defaults = [f"ROI {row + 1}", f"{start:.10g}", f"{stop:.10g}"]
        for column, value in enumerate(defaults):
            self.roi_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self._set_roi_color_button(row, "")
        self.roi_table.blockSignals(False)
        self.roi_table.selectRow(row)
        self._update_preview()

    def _set_roi_color_button(self, row: int, color: str) -> None:
        button = QtWidgets.QPushButton()
        button.setToolTip("Click to choose the ROI shading and curve color.")
        button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        button.setProperty("roiColor", color)
        button.clicked.connect(lambda _checked=False, target=button: self._choose_roi_color(target))
        self.roi_table.setCellWidget(row, 3, button)
        self._style_roi_color_button(button, row)

    def _style_roi_color_button(self, button: QtWidgets.QPushButton, row: int) -> None:
        selected = str(button.property("roiColor") or "").strip()
        effective = selected or ROI_FALLBACK_COLORS[row % len(ROI_FALLBACK_COLORS)]
        qcolor = QtGui.QColor(effective)
        text_color = "#20252b" if qcolor.lightness() > 155 else "#ffffff"
        button.setText(effective.upper())
        button.setStyleSheet(
            f"QPushButton {{ background-color: {effective}; color: {text_color}; font-weight: 600; }}"
        )

    def _choose_roi_color(self, button: QtWidgets.QPushButton) -> None:
        row = next(
            (index for index in range(self.roi_table.rowCount()) if self.roi_table.cellWidget(index, 3) is button),
            -1,
        )
        if row < 0:
            return
        selected = str(button.property("roiColor") or "").strip()
        initial = QtGui.QColor(selected or ROI_FALLBACK_COLORS[row % len(ROI_FALLBACK_COLORS)])
        color = QtWidgets.QColorDialog.getColor(initial, self, "Choose ROI color")
        if not color.isValid():
            return
        button.setProperty("roiColor", color.name(QtGui.QColor.NameFormat.HexRgb))
        self._style_roi_color_button(button, row)
        self._update_preview()

    def _roi_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() not in {0, 1, 2}:
            return
        row = item.row()
        try:
            start = float(self.roi_table.item(row, 1).text())
            stop = float(self.roi_table.item(row, 2).text())
        except (AttributeError, ValueError):
            return
        if np.isfinite(start) and np.isfinite(stop):
            self._update_preview()

    def _roi_enabled_changed(self, enabled: bool) -> None:
        self.roi_pick_button.setEnabled(enabled)
        if not enabled:
            self.roi_pick_button.setChecked(False)
        self._update_preview()

    def _toggle_roi_picker(self, enabled: bool) -> None:
        self._roi_pick_start = None
        self.preview_canvas.setCursor(
            QtCore.Qt.CursorShape.CrossCursor if enabled else QtCore.Qt.CursorShape.ArrowCursor
        )
        if enabled:
            self.preview_caption.setText("ROI selection: click the start position")
        else:
            self._update_preview()

    def _on_preview_click(self, event) -> None:
        if not self.roi_pick_button.isChecked() or event.button != 1 or event.xdata is None:
            return
        if not self.preview_figure.axes or event.inaxes is not self.preview_figure.axes[0]:
            return
        value = float(event.xdata)
        if not np.isfinite(value):
            return
        if self._roi_pick_start is None:
            self._roi_pick_start = value
            self.preview_caption.setText(f"ROI start {value:.6g}: click the stop position")
            self._draw_roi_pick_marker()
            self.preview_canvas.draw_idle()
            return
        start, stop = sorted((self._roi_pick_start, value))
        if np.isclose(start, stop):
            self.preview_caption.setText("ROI start and stop must be different; click the stop position again")
            return
        self._append_roi(start, stop)
        self._roi_pick_start = None
        self.roi_pick_button.setChecked(False)
        self.preview_caption.setText(f"Added ROI {start:.6g} to {stop:.6g}")
        self._update_preview()

    def _draw_roi_pick_marker(self) -> None:
        if self._roi_pick_start is not None and self.preview_figure.axes:
            self.preview_figure.axes[0].axvline(
                self._roi_pick_start,
                color="#d04a3a",
                linewidth=1.2,
                linestyle="--",
            )

    def _remove_roi(self) -> None:
        rows = sorted({index.row() for index in self.roi_table.selectedIndexes()}, reverse=True)
        for row in rows: self.roi_table.removeRow(row)
        for row in range(self.roi_table.rowCount()):
            button = self.roi_table.cellWidget(row, 3)
            if isinstance(button, QtWidgets.QPushButton):
                self._style_roi_color_button(button, row)
        self._update_preview()

    def _read_roi_rows(self) -> list[RoiRange]:
        ranges = []
        for row in range(self.roi_table.rowCount()):
            try:
                name = self.roi_table.item(row, 0).text().strip() or f"ROI {row + 1}"
                start = float(self.roi_table.item(row, 1).text()); stop = float(self.roi_table.item(row, 2).text())
                color_button = self.roi_table.cellWidget(row, 3)
                color = str(color_button.property("roiColor") or "").strip()
            except (AttributeError, ValueError) as exc: raise ValueError(f"ROI row {row + 1} contains invalid values") from exc
            ranges.append(RoiRange(start, stop, color, name))
        return ranges

    def _write_roi_rows(self, ranges: list[RoiRange]) -> None:
        self.roi_table.blockSignals(True)
        self.roi_table.setRowCount(0)
        for roi in ranges:
            row = self.roi_table.rowCount(); self.roi_table.insertRow(row)
            for column, value in enumerate([roi.name, str(roi.start), str(roi.stop)]): self.roi_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
            self._set_roi_color_button(row, roi.color)
        self.roi_table.blockSignals(False)

    def _reset_section(self, section: str) -> None:
        defaults = AppSettings()
        if section == "engine":
            self._set_worker_count(defaults.runtime.workers)
        elif section == "import":
            values = defaults.import_options
            self.auto_header_check.setChecked(values.auto_header)
            self.header_rows_spin.setValue(values.header_rows)
            self._set_combo(self.delimiter_combo, values.delimiter)
            self._set_combo(self.alignment_combo, values.alignment)
            self.recursive_check.setChecked(values.recursive_folders)
        elif section == "data":
            values = defaults.data
            self._set_combo(self.data_type_combo, values.data_type)
            self.range_check.setChecked(values.range_enabled)
            self.range_start_spin.setValue(values.range_start)
            self.range_stop_spin.setValue(values.range_stop)
            self._set_combo(self.input_unit_combo, values.input_unit)
            self._set_combo(self.output_unit_combo, values.output_unit)
            self.wavelength_spin.setValue(values.wavelength_angstrom)
        elif section == "baseline":
            values = defaults.baseline
            self.baseline_check.setChecked(values.enabled)
            self._set_combo(self.algorithm_combo, values.algorithm)
            self.lambda_spin.setValue(values.lam)
            self.asymmetry_spin.setValue(values.asymmetry)
            self.iterations_spin.setValue(values.max_iterations)
            self.tolerance_spin.setValue(values.tolerance)
        elif section == "normalization":
            values = defaults.normalization
            self.ground_check.setChecked(values.ground)
            self._set_combo(self.normalization_combo, values.mode)
        elif section == "reference":
            values = defaults.reference
            self._reference_curve_index = values.curve_index
            self._reference_curve_label = values.curve_label
            self.reference_check.setChecked(values.enabled)
            self.reference_path_edit.setText(values.filename)
            self._update_reference_source_label()
            self.reference_range_check.setChecked(values.fit_range_enabled)
            self.reference_start_spin.setValue(values.fit_range_start)
            self.reference_stop_spin.setValue(values.fit_range_stop)
        elif section == "plot":
            values = defaults.plot
            self.plot_check.setChecked(values.enabled)
            self.colormap_combo.setCurrentText(values.colormap)
            self._x_label_manually_selected = False
            self.x_label_combo.setCurrentText(self._suggested_x_label(self._effective_data_type()))
            self.automatic_time_axis_check.setChecked(values.automatic_time_axis)
            self.scan_interval_spin.setValue(values.scan_interval_seconds)
            self.colorbar_check.setChecked(values.colorbar)
            self.flip_x_data_check.setChecked(values.flip_x_data)
            self.flip_y_data_check.setChecked(values.flip_y_data)
            self.flip_x_axis_check.setChecked(values.flip_x_axis)
            self.flip_y_axis_check.setChecked(values.flip_y_axis)
        elif section == "plot_image":
            values = defaults.plot
            self._set_combo(self.display_scaling_combo, values.display_scaling)
            self.percentile_low_spin.setValue(values.percentile_low)
            self.percentile_high_spin.setValue(values.percentile_high)
            self.value_limits_check.setChecked(values.value_limits_enabled)
            self.value_min_spin.setValue(values.value_min)
            self.value_max_spin.setValue(values.value_max)
            self.width_spin.setValue(values.width_inches)
            self.height_spin.setValue(values.height_inches)
            self.dpi_spin.setValue(values.dpi)
            self._set_combo(self.image_format_combo, values.image_format)
        elif section == "rois":
            values = defaults.rois
            self.roi_check.setChecked(values.enabled)
            self._set_combo(self.roi_selection_combo, values.selection)
            self._set_combo(self.roi_normalization_combo, values.normalization)
            self._write_roi_rows(values.ranges)
        elif section == "output":
            values = defaults.output
            self._last_output_directory = ""
            self._set_combo(self.output_delimiter_combo, values.delimiter)
            self.save_corrected_check.setChecked(values.save_corrected)
            self.save_baseline_check.setChecked(values.save_baseline)
            self.save_original_check.setChecked(values.save_original)
            self.save_reference_check.setChecked(values.save_reference_corrected)
            self.save_plot_check.setChecked(values.save_plot)
            self.save_rois_check.setChecked(values.save_rois)
        else:
            raise ValueError(f"Unknown settings section: {section}")
        labels = {
            "engine": "engine",
            "import": "Import",
            "data": "Data axis and range",
            "baseline": "Background subtraction",
            "normalization": "Post-processing",
            "reference": "Reference subtraction",
            "plot": "Result plot",
            "plot_image": "Color scale and image",
            "rois": "ROIs",
            "output": "Output",
        }
        self._log(f"Reset {labels[section]} to application defaults.")
        if section in {"data", "plot", "plot_image", "rois"}:
            self._update_preview()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths: list[Path] = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir(): paths.extend(self._discover_files_with_feedback(path))
            elif path.suffix.lower() in SUPPORTED_SUFFIXES: paths.append(path)
        self._add_paths(paths); event.acceptProposedAction()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker:
            answer = QtWidgets.QMessageBox.question(self, "Processing is running", "Cancel processing and close the application?")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes: event.ignore(); return
            self._close_after_cancel = True
            self.worker.request_cancel()
            self.status_label.setText("Cancelling before exit…")
            event.ignore()
            return
        event.accept()

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.about(self, "Background Correction Studio", "<b>Background Correction Studio</b><br>Modern, reproducible batch baseline subtraction and plotting.<br><br>Development version 1.0")

    def _show_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Background Correction", message)

    def _log(self, message: str) -> None:
        timestamp = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self.log_view.appendPlainText(f"{timestamp}  {message}")

    @staticmethod
    def _form_container() -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        widget = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(widget); layout.setContentsMargins(10, 10, 10, 10); return widget, layout

    @staticmethod
    def _group_form(title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QFormLayout]:
        group = QtWidgets.QGroupBox(title); form = QtWidgets.QFormLayout(group); form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow); return group, form

    def _add_reset_row(self, form: QtWidgets.QFormLayout, section: str) -> QtWidgets.QPushButton:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addStretch()
        button = QtWidgets.QPushButton("Reset section")
        button.setObjectName(f"reset_{section}_button")
        button.setToolTip("Restore this section to the application defaults without changing other sections.")
        button.clicked.connect(lambda checked=False, name=section: self._reset_section(name))
        layout.addWidget(button)
        form.addRow(row)
        return button

    def _install_wheel_guard(self) -> None:
        for widget in self.findChildren(QtWidgets.QAbstractSpinBox):
            widget.installEventFilter(self)
        for widget in self.findChildren(QtWidgets.QComboBox):
            widget.installEventFilter(self)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if (
            hasattr(self, "file_table")
            and watched is self.file_table.viewport()
            and event.type() == QtCore.QEvent.Type.Resize
        ):
            QtCore.QTimer.singleShot(0, self._fit_file_columns_to_viewport)
        if event.type() == QtCore.QEvent.Type.Wheel and isinstance(
            watched, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)
        ):
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    @staticmethod
    def _scroll_tab(content: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame); scroll.setWidget(content); return scroll

    @staticmethod
    def _float_spin(minimum: float, maximum: float, decimals: int) -> QtWidgets.QDoubleSpinBox:
        spin = CompactDoubleSpinBox(); spin.setRange(minimum, maximum); spin.setDecimals(decimals); spin.setKeyboardTracking(False); spin.setAccelerated(True); return spin

    @staticmethod
    def _combo(items: list[tuple[str, object]]) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        for label, value in items: combo.addItem(label, value)
        return combo

    def _unit_combo(self) -> QtWidgets.QComboBox:
        return self._combo([("Unchanged", "unchanged"), ("q / nm⁻¹", "q_nm"), ("q / Å⁻¹", "q_angstrom"), ("2θ / degrees", "two_theta_deg")])

    @staticmethod
    def _set_combo(combo: QtWidgets.QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0: combo.setCurrentIndex(index)

    def _set_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background: #f4f6f8; color: #20252b; }
            QLabel, QCheckBox, QRadioButton { color: #20252b; }
            QMenuBar { background: #f4f6f8; color: #20252b; }
            QMenuBar::item:selected, QMenu::item:selected { background: #dcebf5; }
            QMenu { background: #ffffff; color: #20252b; border: 1px solid #c8ced5; }
            QTabWidget::pane, QGroupBox { border: 1px solid #d6dbe1; border-radius: 6px; background: #ffffff; color: #20252b; }
            QTabBar::tab { background: #e8ebee; color: #20252b; border: 1px solid #c8ced5; padding: 5px 12px; }
            QTabBar::tab:selected { background: #ffffff; border-bottom-color: #ffffff; }
            QGroupBox { margin-top: 12px; padding: 14px 10px 10px 10px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #20252b; }
            QPushButton { min-height: 26px; padding: 2px 10px; background: #f0f2f4; color: #20252b; border: 1px solid #bcc4cc; border-radius: 3px; }
            QPushButton:hover { background: #e3e8ec; }
            QPushButton:disabled { background: #edf0f2; color: #7b848d; border-color: #d3d8dd; }
            QPushButton#primaryButton { background: #1769aa; color: white; border: none; border-radius: 4px; min-height: 32px; font-weight: 600; padding: 2px 18px; }
            QPushButton#primaryButton:hover { background: #0d5c98; }
            QPushButton#primaryButton:disabled { background: #aeb8c1; color: #f5f6f7; }
            QLabel#panelTitle { font-size: 17px; font-weight: 700; }
            QLabel#engineLabel { font-weight: 700; margin-left: 6px; }
            QLabel#hintLabel { color: #586470; }
            QTableWidget, QPlainTextEdit { background: #ffffff; color: #20252b; alternate-background-color: #f8fafb; gridline-color: #e5e8eb; selection-background-color: #2f8bc2; selection-color: #ffffff; }
            QHeaderView::section { background: #eef1f4; color: #20252b; padding: 5px; border: 0; border-right: 1px solid #d6dbe1; font-weight: 600; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 25px; background: #ffffff; color: #20252b; border: 1px solid #aeb6bf; border-radius: 3px; padding: 0 4px; }
            QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { background: #edf0f2; color: #747d86; border-color: #d2d7dc; }
            QComboBox QAbstractItemView { background: #ffffff; color: #20252b; selection-background-color: #2f8bc2; selection-color: #ffffff; }
            QProgressBar { background: #ffffff; color: #20252b; border: 1px solid #b8c0c8; border-radius: 3px; text-align: center; }
            QProgressBar::chunk { background: #2f8bc2; }
            QScrollArea, QScrollArea > QWidget > QWidget { background: #f4f6f8; }
        """)
