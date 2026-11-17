from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from background_app.config import AppSettings
from background_app.data import DatasetData, LoadedFile
from background_app.processing import process_dataset
from background_app.resources import install_ui_font
from background_app.ui import FileEntry, MainWindow
from background_correction_studio import apply_fixed_light_theme


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / ".tmp" / "ui-preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    apply_fixed_light_theme(application)
    install_ui_font(application)
    settings = AppSettings(); settings.runtime.workers = 1
    window = MainWindow(settings)
    x = np.linspace(2.0, 25.0, 500)
    entries = []
    rows = []
    labels = []
    paths = []
    for file_index in range(3):
        file_rows = []
        for curve_index in range(2):
            background = 50.0 + 0.2 * (x - 12.0) ** 2
            peak = (80 + 12 * file_index + 5 * curve_index) * np.exp(-((x - (8.5 + 1.7 * file_index)) / 0.32) ** 2)
            file_rows.append(background + peak)
        path = Path(f"series_{file_index + 1:02d}.xy")
        loaded = LoadedFile(path, x, np.vstack(file_rows), header=["Synthetic visual QA"])
        entries.append(FileEntry(path=path, loaded=loaded))
        rows.extend(file_rows); labels.extend(loaded.labels); paths.append(path)
    window.entries = entries; window.dataset_name_edit.setText("example_series"); window._refresh_table()
    result = process_dataset(DatasetData(x, np.vstack(rows), labels, paths), settings)
    window.last_result = result; window.preview_mode.setCurrentIndex(2); window._update_preview()
    if len(sys.argv) > 2:
        window.tabs.setCurrentIndex(int(sys.argv[2]))
    window.status_label.setText(f"Complete · {result.elapsed_seconds:.2f} s · {result.workers_used} worker")
    window.progress_bar.setValue(100)
    window.resize(1440, 900); window.show(); application.processEvents()
    if not window.grab().save(str(output), "PNG"):
        raise RuntimeError(f"Could not save {output}")
    print(output)
    window.close(); application.quit(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
