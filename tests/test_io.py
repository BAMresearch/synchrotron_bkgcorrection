import struct
import unittest
from pathlib import Path

import numpy as np

from background_app.config import ImportSettings
from background_app.data import DataValidationError, LoadedFile, combine_files
from background_app.io import load_file, load_text_file
from support import cleanup_scratch, scratch_path


def tearDownModule():
    cleanup_scratch()


class ImportTests(unittest.TestCase):
    def test_reader_keeps_final_row_without_trailing_newline(self):
        path = scratch_path("reader-final-row.xy")
        try:
            path.write_text("header\n1 10\n2 20\n3 30", encoding="utf-8")
            loaded = load_text_file(path, ImportSettings(auto_header=True))
        finally:
            path.unlink(missing_ok=True)
        np.testing.assert_allclose(loaded.x, [1, 2, 3])
        np.testing.assert_allclose(loaded.ys[0], [10, 20, 30])
        self.assertEqual(loaded.header, ["header"])

    def test_auto_detects_comma_delimited_multicurve_data(self):
        path = scratch_path("reader-comma.csv")
        try:
            path.write_text("x,a,b\n1,10,15\n2,20,25\n3,30,35\n", encoding="utf-8")
            loaded = load_text_file(path, ImportSettings(auto_header=True))
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(loaded.detected_delimiter, "comma")
        self.assertEqual(loaded.ys.shape, (2, 3))
        self.assertEqual(loaded.labels, ["a", "b"])

    def test_csv_uses_named_header_over_repeated_units_and_ignores_trailing_comma(self):
        path = scratch_path("reader-raman.csv")
        try:
            path.write_text(
                ",scan 000,scan 001,\nRaman shift [1/cm],Counts,Counts,\n"
                "100,10,15\n200,20,25\n300,30,35\n",
                encoding="utf-8",
            )
            loaded = load_text_file(path, ImportSettings(auto_header=True))
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(loaded.labels, ["scan 000", "scan 001"])
        self.assertEqual(loaded.data_type, "raman")
        self.assertEqual(loaded.x_label, "Raman shift / cm⁻¹")

    def test_native_spc_imports_multiple_float_spectra_and_times(self):
        path = scratch_path("reader-native.spc")
        header = struct.Struct("<cccciddicccci9s9sh32s130s30siicchf48sfifc187s")
        subheader = struct.Struct("<BBHfffIIf4s")
        values = (
            b"\x14", b"K", b"\x0b", b"\x80", 3, 100.0, 300.0, 2,
            b"\x0d", b"\x04", b"\x04", b"\x00", 0, b"", b"", 0,
            b"", b"Raman test", b"", 0, 0, b"\x00", b"\x00", 0, 0.0,
            b"", 0.0, 0, 0.0, b"\x00", b"",
        )
        content = bytearray(header.pack(*values))
        content.extend(subheader.pack(0, 0x80, 1, 0.0, 0.0, 0.0, 3, 0, 0.0, b""))
        content.extend(struct.pack("<3f", 1.0, 2.0, 3.0))
        content.extend(subheader.pack(0, 0x80, 2, 10.0, 0.0, 0.0, 3, 0, 0.0, b""))
        content.extend(struct.pack("<3f", 4.0, 5.0, 6.0))
        try:
            path.write_bytes(content)
            loaded = load_file(path, ImportSettings())
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(loaded.detected_delimiter, "native binary SPC")
        self.assertEqual(loaded.data_type, "raman")
        self.assertEqual(loaded.x_label, "Raman shift / cm⁻¹")
        np.testing.assert_allclose(loaded.x, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(loaded.ys, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(loaded.labels, ["reader-native.spc[0 s]", "reader-native.spc[10 s]"])

    def test_pxrd_header_selects_a_two_theta_axis(self):
        path = scratch_path("reader-pxrd.xy")
        try:
            path.write_text("2theta intensity\n5 10\n10 20\n15 30\n", encoding="utf-8")
            loaded = load_text_file(path, ImportSettings(auto_header=True))
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(loaded.data_type, "pxrd")
        self.assertEqual(loaded.x_label, "2θ / °")

    def test_inconsistent_columns_report_the_line(self):
        path = scratch_path("reader-broken.xy")
        try:
            path.write_text("1 10\n2 20 30\n3 30\n", encoding="utf-8")
            with self.assertRaisesRegex(DataValidationError, "found 3 columns"):
                load_text_file(path, ImportSettings(auto_header=True))
        finally:
            path.unlink(missing_ok=True)

    def test_interpolation_alignment_is_explicit(self):
        first = LoadedFile(Path("a.xy"), np.array([0.0, 1.0, 2.0]), np.array([[0.0, 1.0, 2.0]]))
        second = LoadedFile(Path("b.xy"), np.array([0.0, 0.5, 1.0, 1.5, 2.0]), np.array([[0.0, 1.0, 2.0, 3.0, 4.0]]))
        with self.assertRaises(DataValidationError):
            combine_files([first, second], "strict")
        combined = combine_files([first, second], "interpolate")
        np.testing.assert_allclose(combined.ys[1], [0.0, 2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
