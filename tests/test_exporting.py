import unittest
from pathlib import Path

import numpy as np

from background_app.config import AppSettings
from background_app.data import DatasetData
from background_app.exporting import export_result
from background_app.processing import process_dataset
from support import cleanup_scratch, scratch_path


def tearDownModule():
    cleanup_scratch()


class ExportTests(unittest.TestCase):
    def test_export_writes_data_plot_and_manifest(self):
        x = np.linspace(0.0, 10.0, 100)
        y = 2.0 + np.sin(x) ** 2
        dataset = DatasetData(x, np.vstack([y, y * 1.2]), ["a", "b"], [Path("input.xy")])
        settings = AppSettings(); settings.runtime.workers = 1; settings.data.range_enabled = False
        result = process_dataset(dataset, settings)
        folder = scratch_path("placeholder").parent
        manifest = export_result(result, settings, folder, "unit-test-series")
        try:
            names = {path.name for path in manifest.files}
            self.assertIn("unit-test-series_corrected.dat", names)
            self.assertIn("unit-test-series_baseline.dat", names)
            self.assertIn("unit-test-series.png", names)
            self.assertIn("unit-test-series_run.json", names)
            self.assertTrue(all(path.exists() for path in manifest.files))
        finally:
            for path in manifest.files:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
