import unittest

import numpy as np
from matplotlib.figure import Figure

from pathlib import Path

from background_app.config import AppSettings, PlotSettings
from background_app.data import DatasetData
from background_app.plotting import (
    create_result_figure,
    draw_curve_preview,
    prepare_heatmap_display,
    series_axis_geometry,
)
from background_app.processing import process_dataset


class PlottingTests(unittest.TestCase):
    def test_curve_preview_has_exact_horizontal_data_limits(self):
        figure = Figure()
        x = np.linspace(5.0, 100.0, 50)

        draw_curve_preview(figure, x, np.sin(x))

        np.testing.assert_allclose(figure.axes[0].get_xlim(), [5.0, 100.0])

    def test_heatmap_has_exact_horizontal_data_limits(self):
        x = np.linspace(5.0, 100.0, 50)
        ys = np.vstack([np.sin(x), np.cos(x)])
        dataset = DatasetData(x, ys, ["a", "b"], [Path("input.dat")])
        settings = AppSettings(); settings.runtime.workers = 1
        result = process_dataset(dataset, settings)

        figure = create_result_figure(result, settings.plot)

        np.testing.assert_allclose(figure.axes[0].get_xlim(), [5.0, 100.0])

    def test_automatic_axis_converts_30_second_scans_to_minutes(self):
        settings = PlotSettings(automatic_time_axis=True, scan_interval_seconds=30.0)

        start, stop, positions, label = series_axis_geometry(settings, 180)

        self.assertEqual(label, "Time / min")
        self.assertEqual((start, stop), (0.0, 90.0))
        np.testing.assert_allclose(positions[[0, -1]], [0.25, 89.75])

    def test_manual_axis_uses_one_based_scan_numbers(self):
        settings = PlotSettings(automatic_time_axis=False)

        start, stop, positions, label = series_axis_geometry(settings, 180)

        self.assertEqual(label, "Scan number")
        self.assertEqual((start, stop), (0.5, 180.5))
        np.testing.assert_array_equal(positions[[0, -1]], [1.0, 180.0])

    def test_robust_heatmap_limits_ignore_a_single_extreme_peak(self):
        data = np.concatenate([np.arange(100.0), [1_000_000.0]])[None, :]
        settings = PlotSettings(
            display_scaling="robust",
            percentile_low=1.0,
            percentile_high=99.0,
        )

        displayed, kwargs, label = prepare_heatmap_display(data, settings)

        np.testing.assert_array_equal(displayed, data)
        self.assertLess(kwargs["vmax"], 1_000.0)
        self.assertEqual(label, "Intensity")

    def test_asinh_heatmap_compresses_large_peaks_without_altering_data(self):
        data = np.array([[0.0, 10.0, 1000.0]])
        settings = PlotSettings(
            display_scaling="asinh",
            percentile_low=0.0,
            percentile_high=100.0,
        )

        displayed, kwargs, label = prepare_heatmap_display(data, settings)
        normalized = kwargs["norm"](data[0])

        np.testing.assert_array_equal(displayed, data)
        self.assertGreater(normalized[1] - normalized[0], 0.01)
        self.assertEqual(label, "Intensity (asinh/HDR display)")

    def test_per_scan_scaling_reveals_equal_relative_shapes(self):
        data = np.array([[0.0, 1.0, 2.0], [0.0, 100.0, 200.0]])
        settings = PlotSettings(
            display_scaling="per_scan",
            percentile_low=0.0,
            percentile_high=100.0,
        )

        displayed, kwargs, label = prepare_heatmap_display(data, settings)

        np.testing.assert_allclose(displayed[0], displayed[1])
        np.testing.assert_allclose(displayed[0], [0.0, 0.5, 1.0])
        self.assertEqual(kwargs, {"vmin": 0.0, "vmax": 1.0})
        self.assertEqual(label, "Relative intensity per scan")


if __name__ == "__main__":
    unittest.main()
