import unittest
from pathlib import Path

import numpy as np

from background_app.config import AppSettings, RoiRange
from background_app.data import DatasetData, LoadedFile
from background_app.processing import convert_axis, process_dataset, resolve_worker_count


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(2.0, 25.0, 500)
        background = 50.0 + 0.2 * (self.x - 12.0) ** 2
        signal = background + 100.0 * np.exp(-((self.x - 9.0) / 0.3) ** 2)
        self.dataset = DatasetData(self.x, np.vstack([signal, signal * 1.1]), ["a", "b"], [Path("input.xy")])

    def test_area_normalized_batch_is_finite(self):
        settings = AppSettings(); settings.runtime.workers = 1
        result = process_dataset(self.dataset, settings)
        self.assertEqual(result.final.shape, (2, 500))
        self.assertTrue(np.isfinite(result.final).all())
        areas = np.abs(np.trapezoid(result.final, x=result.x, axis=1))
        np.testing.assert_allclose(areas, [1.0, 1.0], atol=1e-10)

    def test_roi_fractions_sum_to_one(self):
        settings = AppSettings(); settings.runtime.workers = 1
        settings.rois.enabled = True
        settings.rois.ranges = [RoiRange(8.0, 10.0, name="peak"), RoiRange(15.0, 20.0, name="tail")]
        result = process_dataset(self.dataset, settings)
        np.testing.assert_allclose(np.sum(result.roi_values, axis=0), [1.0, 1.0], atol=1e-10)

    def test_q_unit_conversion_and_two_theta_round_trip(self):
        q_nm = np.array([1.0, 2.0, 3.0])
        q_angstrom = convert_axis(q_nm, "q_nm", "q_angstrom", 1.5406)
        np.testing.assert_allclose(q_angstrom, q_nm / 10.0)
        two_theta = convert_axis(q_angstrom, "q_angstrom", "two_theta_deg", 1.5406)
        round_trip = convert_axis(two_theta, "two_theta_deg", "q_angstrom", 1.5406)
        np.testing.assert_allclose(round_trip, q_angstrom)

    def test_auto_workers_avoid_processes_for_small_jobs(self):
        self.assertEqual(resolve_worker_count(0, curves=5, points=500), 1)
        self.assertGreaterEqual(resolve_worker_count(0, curves=100, points=500), 2)

    def test_total_subtracted_matches_correction_with_reference(self):
        settings = AppSettings(); settings.runtime.workers = 1
        settings.normalization.mode = "none"
        settings.reference.enabled = True
        settings.reference.fit_range_enabled = True
        settings.reference.fit_range_start = 8.0
        settings.reference.fit_range_stop = 10.0
        reference_signal = 5.0 + 30.0 * np.exp(-((self.x - 9.0) / 0.3) ** 2)
        reference = LoadedFile(Path("jar.dat"), self.x, reference_signal)

        result = process_dataset(self.dataset, settings, reference_file=reference)

        np.testing.assert_allclose(
            result.original - result.total_subtracted,
            result.corrected,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertFalse(np.allclose(result.total_subtracted, result.baselines))


if __name__ == "__main__":
    unittest.main()
