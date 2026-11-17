import unittest

import numpy as np

from BackgroundCorrection.algorithm import als as legacy_als
from background_app.baseline import BaselineError, fit_baseline
from background_app.config import BaselineSettings


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0.0, 30.0, 500)
        self.background = 20.0 + 0.03 * (self.x - 14.0) ** 2
        self.y = self.background + 90.0 * np.exp(-((self.x - 9.0) / 0.4) ** 2)

    def test_als_matches_legacy_algorithm(self):
        settings = BaselineSettings(algorithm="als", lam=2000.0, asymmetry=0.02, max_iterations=100, tolerance=0.0)
        expected = legacy_als(self.y, 2000.0, 0.02, 100)
        actual = fit_baseline(self.y, settings)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)

    def test_arpls_returns_finite_baseline(self):
        settings = BaselineSettings(algorithm="arpls", lam=2000.0, asymmetry=0.02, max_iterations=50)
        actual = fit_baseline(self.y, settings)
        self.assertTrue(np.isfinite(actual).all())

    def test_invalid_asymmetry_is_rejected(self):
        with self.assertRaises(BaselineError):
            fit_baseline(self.y, BaselineSettings(asymmetry=0.0))


if __name__ == "__main__":
    unittest.main()
