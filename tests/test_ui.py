import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from background_app.config import AppSettings
from background_app.data import LoadedFile, combine_files
from background_app.ui import FileEntry, MainWindow, _load_reference_source


class UiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_main_window_constructs_and_round_trips_controls(self):
        settings = AppSettings(); settings.baseline.lam = 4321.0; settings.plot.colormap = "viridis"
        window = MainWindow(settings)
        current = window.current_settings()
        self.assertEqual(window.tabs.count(), 5)
        self.assertEqual(current.baseline.lam, 4321.0)
        self.assertEqual(current.plot.colormap, "viridis")
        window.close()

    def test_data_type_auto_detection_adapts_pxrd_controls_and_axis_label(self):
        settings = AppSettings()
        settings.data.input_unit = "q_nm"
        settings.data.output_unit = "two_theta_deg"
        window = MainWindow(settings)
        x = np.linspace(100.0, 3200.0, 100)
        loaded = LoadedFile(
            Path("raman.spc"),
            x,
            np.sin(x),
            data_type="raman",
            x_label="Raman shift / cm⁻¹",
        )
        window.entries = [FileEntry(loaded.path, loaded=loaded)]
        window._refresh_table()

        self.assertEqual(window.data_type_combo.currentData(), "auto")
        self.assertIn("Raman detected", window.data_type_combo.itemText(0))
        self.assertTrue(window.input_unit_combo.isHidden())
        self.assertTrue(window.output_unit_combo.isHidden())
        self.assertTrue(window.wavelength_spin.isHidden())
        self.assertEqual(window.current_settings().data.input_unit, "unchanged")
        self.assertEqual(window.current_settings().data.output_unit, "unchanged")
        self.assertEqual(window.x_label_combo.currentText(), "Raman shift / cm⁻¹")
        self.assertGreaterEqual(window.x_label_combo.findText("q / Å⁻¹"), 0)

        window.data_type_combo.setCurrentIndex(window.data_type_combo.findData("pxrd"))
        self.assertFalse(window.input_unit_combo.isHidden())
        self.assertFalse(window.output_unit_combo.isHidden())
        self.assertFalse(window.wavelength_spin.isHidden())
        window.close()

    def test_section_reset_and_numeric_steps_are_field_specific(self):
        window = MainWindow(AppSettings())
        window.lambda_spin.setValue(9876.0)
        window.asymmetry_spin.setValue(0.4)
        window._reset_section("baseline")
        self.assertEqual(window.lambda_spin.value(), 2000.0)
        self.assertAlmostEqual(window.asymmetry_spin.value(), 0.02)
        self.assertEqual(window.lambda_spin.singleStep(), 1.0)
        self.assertAlmostEqual(window.asymmetry_spin.singleStep(), 0.001)
        self.assertAlmostEqual(window.tolerance_spin.singleStep(), 0.0000001)
        self.assertEqual(window.lambda_spin.textFromValue(2000.0), "2000")
        self.assertFalse(window.tolerance_spin.textFromValue(0.000001).endswith("0"))
        self.assertTrue(window.tolerance_spin.toolTip())
        self.assertIsNotNone(window.findChild(QtWidgets.QPushButton, "reset_import_button"))
        window.close()

    def test_two_preview_clicks_create_an_roi(self):
        window = MainWindow(AppSettings())
        window.roi_check.setChecked(True)
        window.roi_pick_button.setChecked(True)
        axis = window.preview_figure.axes[0]
        window._on_preview_click(SimpleNamespace(button=1, xdata=0.2, inaxes=axis))
        window._on_preview_click(SimpleNamespace(button=1, xdata=0.8, inaxes=axis))
        self.assertEqual(window.roi_table.rowCount(), 1)
        self.assertAlmostEqual(float(window.roi_table.item(0, 1).text()), 0.2)
        self.assertAlmostEqual(float(window.roi_table.item(0, 2).text()), 0.8)
        self.assertFalse(window.roi_pick_button.isChecked())
        window.close()

    def test_manual_roi_edits_refresh_shading_and_color_button_opens_picker(self):
        window = MainWindow(AppSettings())
        x = np.linspace(0.0, 10.0, 100)
        loaded = LoadedFile(Path("roi.dat"), x, np.sin(x))
        window.entries = [FileEntry(loaded.path, loaded=loaded)]
        window._refresh_table()
        window.roi_check.setChecked(True)
        window._add_roi()

        window.roi_table.item(0, 1).setText("2")
        window.roi_table.item(0, 2).setText("4")

        self.assertEqual(len(window.preview_figure.axes[0].patches), 1)
        roi = window.current_settings().rois.ranges[0]
        self.assertEqual((roi.start, roi.stop), (2.0, 4.0))
        color_button = window.roi_table.cellWidget(0, 3)
        self.assertIsInstance(color_button, QtWidgets.QPushButton)
        with patch.object(QtWidgets.QColorDialog, "getColor", return_value=QtGui.QColor("#123456")):
            color_button.click()
        self.assertEqual(window.current_settings().rois.ranges[0].color, "#123456")
        self.assertIn("#123456", color_button.text())
        window.close()

    def test_worker_setting_lives_in_menu_and_wheel_changes_are_blocked(self):
        window = MainWindow(AppSettings())
        window._set_worker_count(2)
        self.assertEqual(window.current_settings().runtime.workers, 2)
        self.assertTrue(window.worker_actions[2].isChecked())
        wheel = QtCore.QEvent(QtCore.QEvent.Type.Wheel)
        self.assertTrue(window.eventFilter(window.lambda_spin, wheel))
        window.close()

    def test_live_preview_draws_baseline_and_corrected_curve(self):
        window = MainWindow(AppSettings())
        x = np.linspace(2.0, 47.0, 100)
        y = 10.0 + 0.01 * (x - 20.0) ** 2 + 20.0 * np.exp(-((x - 18.0) / 0.8) ** 2)
        loaded = LoadedFile(Path("live.dat"), x, y)
        window.entries = [FileEntry(loaded.path, loaded=loaded)]
        window._refresh_table()
        window._render_live_preview()
        self.assertGreaterEqual(len(window.preview_figure.axes[0].lines), 3)
        self.assertIn("Live", window.preview_caption.text())
        window.close()

    def test_file_rows_reorder_the_backing_series_and_table_is_simplified(self):
        window = MainWindow(AppSettings())
        x = np.linspace(2.0, 47.0, 20)
        window.entries = [
            FileEntry(Path(name), loaded=LoadedFile(Path(name), x, np.sin(x)))
            for name in ("a.dat", "b.dat", "c.dat")
        ]
        window._refresh_table()
        window._reorder_entry(0, 2)
        self.assertEqual([entry.path.name for entry in window.entries], ["b.dat", "c.dat", "a.dat"])
        self.assertEqual(window.file_table.columnCount(), 5)
        self.assertEqual(
            [window.file_table.horizontalHeaderItem(column).text() for column in range(5)],
            ["Use", "Data item", "Points", "x min", "x max"],
        )
        self.assertTrue(
            all(
                window.file_table.horizontalHeader().sectionResizeMode(column)
                == QtWidgets.QHeaderView.ResizeMode.Interactive
                for column in (0, 2, 3)
            )
        )
        self.assertEqual(
            window.file_table.horizontalHeader().sectionResizeMode(1),
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.assertEqual(
            window.file_table.horizontalHeader().sectionResizeMode(4),
            QtWidgets.QHeaderView.ResizeMode.Fixed,
        )
        self.assertEqual(
            window.file_table.horizontalScrollBarPolicy(),
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        window.resize(1180, 760); window.show(); self.app.processEvents()
        header = window.file_table.horizontalHeader()
        self.assertGreater(header.sectionSize(1), max(header.sectionSize(column) for column in (0, 2, 3, 4)))
        header.resizeSection(2, window.file_table.viewport().width())
        self.app.processEvents()
        self.assertLessEqual(header.length(), window.file_table.viewport().width() + 1)
        self.assertTrue(window.file_table.dragEnabled())
        window.close()

    def test_import_advances_the_shared_progress_bar_for_each_file(self):
        window = MainWindow(AppSettings())
        window.resize(1180, 760); window.show(); self.app.processEvents()
        x = np.linspace(2.0, 47.0, 20)
        progress_values = []
        window.progress_bar.valueChanged.connect(progress_values.append)

        def loaded(path, _settings):
            return LoadedFile(Path(path), x, np.sin(x))

        with patch("background_app.ui.load_file", side_effect=loaded):
            window._add_paths([Path("first.dat"), Path("second.dat")])

        self.assertIn(50, progress_values)
        self.assertEqual(window.progress_bar.value(), 100)
        self.assertIn("Import complete", window.status_label.text())
        self.assertEqual(len(window.entries), 2)
        self.app.processEvents()
        self.assertLessEqual(
            window.file_table.horizontalHeader().length(),
            window.file_table.viewport().width() + 1,
        )
        window.close()

    def test_multispectrum_spc_and_csv_are_individual_reorderable_items(self):
        for suffix in (".spc", ".csv"):
            with self.subTest(suffix=suffix):
                window = MainWindow(AppSettings())
                x = np.linspace(100.0, 300.0, 20)
                path = Path(f"series{suffix}")
                loaded = LoadedFile(
                    path,
                    x,
                    np.vstack([np.sin(x), np.cos(x), np.sin(x) * 2.0]),
                    labels=["scan 000", "scan 001", "scan 002"],
                )
                with patch("background_app.ui.load_file", return_value=loaded) as reader:
                    window._add_paths([path])

                reader.assert_called_once()
                self.assertEqual(window.file_table.rowCount(), 3)
                self.assertEqual([entry.curve_index for entry in window.entries], [0, 1, 2])
                self.assertEqual([entry.loaded.curves for entry in window.entries], [1, 1, 1])
                self.assertEqual(
                    [window.file_table.item(row, 1).text() for row in range(3)],
                    ["scan 000", "scan 001", "scan 002"],
                )
                window.file_table.selectRow(1)
                window._update_preview()
                self.assertIn("scan 001", window.preview_figure.axes[0].get_title())
                self.assertTrue(window.curve_spin.isHidden())
                window._reorder_entry(1, 0)
                self.assertEqual([entry.display_name for entry in window.entries], ["scan 001", "scan 000", "scan 002"])
                combined = combine_files([entry.loaded for entry in window.entries])
                self.assertEqual(combined.labels, ["scan 001", "scan 000", "scan 002"])
                self.assertEqual(combined.source_paths, [path])
                with patch("background_app.ui.load_file", return_value=loaded) as reload_reader:
                    window._reload_entries()
                reload_reader.assert_called_once()
                self.assertEqual([entry.display_name for entry in window.entries], ["scan 001", "scan 000", "scan 002"])
                window.close()

    def test_embedded_spc_item_can_be_dropped_as_the_reference(self):
        window = MainWindow(AppSettings())
        x = np.linspace(100.0, 300.0, 20)
        path = Path("raman-series.spc")
        loaded = LoadedFile(
            path,
            x,
            np.vstack([np.sin(x), np.cos(x), np.sin(x) * 2.0]),
            labels=["sample 001", "empty holder", "sample 002"],
        )
        with patch("background_app.ui.load_file", return_value=loaded):
            window._add_paths([path])

        item = window.file_table.item(1, 1)
        mime_data = window.file_table.mimeData([item])
        self.assertTrue(mime_data.hasFormat("application/x-background-correction-reference-item"))
        drop_event = QtGui.QDropEvent(
            QtCore.QPointF(5.0, 5.0),
            QtCore.Qt.DropAction.CopyAction,
            mime_data,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        window.reference_path_edit.dropEvent(drop_event)

        settings = window.current_settings()
        self.assertTrue(settings.reference.enabled)
        self.assertEqual(settings.reference.filename, str(path))
        self.assertEqual(settings.reference.curve_index, 1)
        self.assertEqual(settings.reference.curve_label, "empty holder")
        self.assertFalse(window.entries[1].enabled)
        self.assertEqual(
            window.file_table.item(1, 0).checkState(),
            QtCore.Qt.CheckState.Unchecked,
        )
        self.assertIn("empty holder", window.reference_source_label.text())

        with patch("background_app.ui.load_file", return_value=loaded):
            reference = _load_reference_source(settings)
        self.assertEqual(reference.curves, 1)
        np.testing.assert_allclose(reference.ys[0], loaded.ys[1])
        window.close()

    def test_row_drop_moves_the_dragged_item_after_the_target(self):
        window = MainWindow(AppSettings())
        x = np.linspace(0, 10, 20)
        for name in ("first.dat", "second.dat", "third.dat"):
            path = Path(name)
            window.entries.append(FileEntry(path, loaded=LoadedFile(path, x, x)))
        window._refresh_table()
        table = window.file_table
        mime = table.mimeData([table.item(0, 1)])
        table.selectRow(1)
        target = table.visualRect(table.model().index(2, 1)).bottomLeft()
        event = unittest.mock.Mock()
        event.source.return_value = table
        event.position.return_value = QtCore.QPointF(target)
        event.mimeData.return_value = mime
        table.dropEvent(event)
        self.assertEqual([entry.path.name for entry in window.entries],
                         ["second.dat", "third.dat", "first.dat"])
        event.accept.assert_called_once()
        window.close()

    def test_preview_can_show_the_two_removed_components(self):
        settings = AppSettings()
        window = MainWindow(settings)
        x = np.linspace(0, 10, 20)
        result = SimpleNamespace(
            x=x, original=np.array([4 * x]), corrected=np.array([x]),
            final=np.array([x]), baselines=np.array([x]),
            reference_scaled=np.array([2 * x]), reference_scales=np.array([2]),
            labels=["scan"],
        )
        window.split_reference_check.setChecked(True)
        window._draw_processed_curve(result, 0, settings)
        lines = {line.get_label(): line.get_ydata() for line in window.preview_figure.axes[0].lines}
        np.testing.assert_array_equal(lines["Scaled JAR/reference"], 2 * x)
        np.testing.assert_array_equal(lines["Additional fitted baseline"], x)
        window.close()

    def test_roi_picker_is_in_roi_tab_and_tolerance_field_is_wide(self):
        window = MainWindow(AppSettings())
        roi_tab = window.tabs.widget(3)
        self.assertTrue(roi_tab.isAncestorOf(window.roi_pick_button))
        self.assertFalse(window.preview_panel.isAncestorOf(window.roi_pick_button))
        self.assertGreaterEqual(window.tolerance_spin.minimumWidth(), 180)
        window.close()

    def test_plot_uses_automatic_minute_axis_instead_of_free_text_label(self):
        window = MainWindow(AppSettings())
        self.assertTrue(window.automatic_time_axis_check.isChecked())
        self.assertEqual(window.scan_interval_spin.value(), 30.0)
        self.assertFalse(hasattr(window, "y_label_edit"))
        settings = window.current_settings()
        self.assertTrue(settings.plot.automatic_time_axis)
        self.assertEqual(settings.plot.scan_interval_seconds, 30.0)
        window.automatic_time_axis_check.setChecked(False)
        self.assertFalse(window.scan_interval_spin.isEnabled())
        window.close()

    def test_heatmap_scaling_defaults_to_robust_and_enables_relevant_controls(self):
        window = MainWindow(AppSettings())

        self.assertEqual(window.display_scaling_combo.currentData(), "robust")
        self.assertTrue(window.percentile_low_spin.isEnabled())
        self.assertTrue(window.percentile_high_spin.isEnabled())
        self.assertFalse(window.value_limits_check.isEnabled())
        self.assertAlmostEqual(window.percentile_low_spin.value(), 1.0)
        self.assertAlmostEqual(window.percentile_high_spin.value(), 99.5)

        window.display_scaling_combo.setCurrentIndex(
            window.display_scaling_combo.findData("linear")
        )
        self.assertFalse(window.percentile_low_spin.isEnabled())
        self.assertFalse(window.percentile_high_spin.isEnabled())
        self.assertTrue(window.value_limits_check.isEnabled())
        window.value_limits_check.setChecked(True)
        self.assertTrue(window.value_min_spin.isEnabled())
        self.assertTrue(window.value_max_spin.isEnabled())
        self.assertEqual(window.current_settings().plot.display_scaling, "linear")
        window.close()

    def test_run_is_cancelled_cleanly_when_no_destination_is_selected(self):
        window = MainWindow(AppSettings())
        x = np.linspace(2.0, 47.0, 20)
        loaded = LoadedFile(Path("input.dat"), x, np.sin(x))
        window.entries = [FileEntry(loaded.path, loaded=loaded)]
        with patch.object(QtWidgets.QFileDialog, "getExistingDirectory", return_value=""):
            window._start_processing()
        self.assertIsNone(window.worker)
        self.assertIn("no output destination", window.log_view.toPlainText())
        window.close()


if __name__ == "__main__":
    unittest.main()
