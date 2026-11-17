import json
import unittest
from pathlib import Path

from background_app.config import AppSettings, RoiRange
from support import cleanup_scratch, scratch_path


def tearDownModule():
    cleanup_scratch()


class ConfigTests(unittest.TestCase):
    def test_x_range_restriction_is_off_by_default(self):
        self.assertFalse(AppSettings().data.range_enabled)

    def test_round_trip_preserves_nested_settings(self):
        settings = AppSettings()
        settings.baseline.algorithm = "arpls"
        settings.data.data_type = "raman"
        settings.runtime.workers = 4
        settings.plot.scan_interval_seconds = 12.5
        settings.plot.display_scaling = "asinh"
        settings.plot.percentile_high = 98.5
        settings.reference.curve_index = 7
        settings.reference.curve_label = "background scan"
        settings.rois.enabled = True
        settings.rois.ranges = [RoiRange(2.0, 4.0, "red", "peak")]
        path = scratch_path("roundtrip-settings.json")
        try:
            settings.save(path)
            loaded = AppSettings.load(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(loaded.baseline.algorithm, "arpls")
        self.assertEqual(loaded.data.data_type, "raman")
        self.assertEqual(loaded.runtime.workers, 4)
        self.assertEqual(loaded.plot.scan_interval_seconds, 12.5)
        self.assertEqual(loaded.plot.display_scaling, "asinh")
        self.assertEqual(loaded.plot.percentile_high, 98.5)
        self.assertEqual(loaded.reference.curve_index, 7)
        self.assertEqual(loaded.reference.curve_label, "background scan")
        self.assertEqual(loaded.rois.ranges[0].name, "peak")

    def test_legacy_settings_are_migrated_without_rewriting_source(self):
        legacy = {
            "parallel": {"enable": True, "cores": "auto"},
            "io": {"out_dir": "legacy-output", "dat_file_sep": "\t", "head_row_count": 23},
            "data": {"range_start": 2, "range_stop": 47, "unit_in": 1, "unit_out": 1},
            "baseline": {"enable": True, "algorithm": 1, "itermax": 100, "lam": 2000, "ratio": 0.02},
            "normalization": {"area": True, "sum": False, "max": False, "ground": False},
            "jar": {"enable": False},
            "rois": {"enable": False, "ranges": []},
            "plot": {"enable": True, "time_step": 1, "heatmap": "hot", "x_unit": "q", "y_unit": "s"},
        }
        path = scratch_path("legacy-settings.json")
        try:
            original = json.dumps(legacy)
            path.write_text(original, encoding="utf-8")
            loaded = AppSettings.load(path)
            after = path.read_text(encoding="utf-8")
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(after, original)
        self.assertEqual(loaded.baseline.algorithm, "als")
        self.assertEqual(loaded.import_options.header_rows, 23)
        self.assertEqual(loaded.output.directory, "legacy-output")
        self.assertFalse(loaded.data.range_enabled)
        self.assertTrue(loaded.plot.automatic_time_axis)
        self.assertEqual(loaded.plot.scan_interval_seconds, 1.0)

    def test_previous_minute_axis_settings_are_converted_to_seconds_per_scan(self):
        raw = AppSettings().to_dict()
        raw["plot"].pop("automatic_time_axis")
        raw["plot"].pop("scan_interval_seconds")
        raw["plot"].update({"y_label": "Time / min", "time_step": 0.5})

        loaded = AppSettings.from_dict(raw)

        self.assertTrue(loaded.plot.automatic_time_axis)
        self.assertEqual(loaded.plot.scan_interval_seconds, 30.0)


if __name__ == "__main__":
    unittest.main()
