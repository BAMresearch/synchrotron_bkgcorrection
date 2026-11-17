from __future__ import annotations

import csv
import re
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import ImportSettings
from .data import DataValidationError, LoadedFile


TEXT_SUFFIXES = {".xy", ".dat", ".chi", ".txt", ".raman", ".csv", ".tsv"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".spc"}

_DELIMITERS: dict[str, str | None] = {
    "whitespace": None,
    "tab": "\t",
    "comma": ",",
    "semicolon": ";",
}

_SPC_HEADER = struct.Struct("<cccciddicccci9s9sh32s130s30siicchf48sfifc187s")
_SPC_SUBHEADER = struct.Struct("<BBHfffIIf4s")
_SPC_HEADER_SIZE = 512
_SPC_TSPREC = 0x01
_SPC_TMULTI = 0x04
_SPC_TRANDM = 0x08
_SPC_TXYXYS = 0x40
_SPC_TXVALS = 0x80


def discover_files(folder: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(folder)
    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def load_file(filename: str | Path, settings: ImportSettings) -> LoadedFile:
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return load_text_file(path, settings)
    if suffix == ".spc":
        return load_spc_file(path)
    raise DataValidationError(f"{path.name}: unsupported file type '{suffix}'")


def load_text_file(filename: str | Path, settings: ImportSettings) -> LoadedFile:
    path = Path(filename)
    text = _read_text(path)
    lines = text.splitlines()
    if not lines:
        raise DataValidationError(f"{path.name}: file is empty")

    if settings.auto_header:
        start, delimiter_name = _detect_data_start(lines, settings.delimiter)
    else:
        start = max(0, int(settings.header_rows))
        if start >= len(lines):
            raise DataValidationError(f"{path.name}: header row count exceeds the file length")
        delimiter_name = _detect_delimiter(lines[start:], settings.delimiter)

    delimiter = _DELIMITERS[delimiter_name]
    rows: list[list[float]] = []
    expected_columns: int | None = None
    for line_number, raw_line in enumerate(lines[start:], start=start + 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = _parse_numeric_line(stripped, delimiter)
        if values is None:
            raise DataValidationError(f"{path.name}:{line_number}: expected a numeric data row")
        if expected_columns is None:
            expected_columns = len(values)
        elif len(values) != expected_columns:
            raise DataValidationError(
                f"{path.name}:{line_number}: found {len(values)} columns; expected {expected_columns}"
            )
        rows.append(values)

    if not rows or expected_columns is None or expected_columns < 2:
        raise DataValidationError(f"{path.name}: no numeric x/y data with at least two columns was found")

    data = np.asarray(rows, dtype=np.float64).T
    labels = _extract_column_labels(lines[:start], delimiter_name, expected_columns, path.name)
    data_type, x_label = _detect_text_axis(path, lines[:start], data[0])
    return LoadedFile(
        path=path,
        x=data[0],
        ys=data[1:],
        header=lines[:start],
        labels=labels,
        detected_delimiter=delimiter_name,
        data_type=data_type,
        x_label=x_label,
    )


def load_spc_file(filename: str | Path) -> LoadedFile:
    """Load modern little-endian Galactic SPC files without third-party packages."""
    path = Path(filename)
    try:
        content = path.read_bytes()
        if len(content) < _SPC_HEADER_SIZE:
            raise DataValidationError(f"{path.name}: file is too short to contain an SPC header")
        values = _SPC_HEADER.unpack_from(content)
        flags = values[0][0]
        version = values[1][0]
        global_exponent = values[3][0]
        point_count = int(values[4])
        first_x = float(values[5])
        last_x = float(values[6])
        spectrum_count = int(values[7])
        x_type = values[8][0]
        comment = values[17].split(b"\0", 1)[0].decode("cp1252", errors="replace").strip()

        if version != 0x4B:
            raise DataValidationError(
                f"{path.name}: SPC version 0x{version:02X} is not supported; export modern little-endian SPC or CSV"
            )
        if point_count < 3 or spectrum_count < 1:
            raise DataValidationError(f"{path.name}: invalid SPC dimensions ({spectrum_count} spectra x {point_count} points)")
        if flags & _SPC_TRANDM:
            raise DataValidationError(f"{path.name}: random-access SPC subfile directories are not supported")

        offset = _SPC_HEADER_SIZE
        if flags & _SPC_TXVALS:
            x, offset = _read_spc_values(content, offset, point_count, np.dtype("<f4"), path.name, "x-axis")
        else:
            x = np.linspace(first_x, last_x, point_count, dtype=np.float64)

        separate_x = bool(flags & _SPC_TXYXYS)
        multi_exponent = bool(flags & _SPC_TMULTI)
        precision_16 = bool(flags & _SPC_TSPREC)
        spectra: list[np.ndarray] = []
        labels: list[str] = []
        shared_x = np.asarray(x, dtype=np.float64)

        for index in range(spectrum_count):
            if offset + _SPC_SUBHEADER.size > len(content):
                raise DataValidationError(f"{path.name}: truncated before SPC spectrum {index + 1}")
            subheader = _SPC_SUBHEADER.unpack_from(content, offset)
            offset += _SPC_SUBHEADER.size
            sub_exponent = int(subheader[1])
            sub_index = int(subheader[2])
            sub_time = float(subheader[3])
            sub_points = int(subheader[6])
            count = sub_points if separate_x and sub_points > 0 else point_count

            if separate_x:
                spectrum_x, offset = _read_spc_values(
                    content, offset, count, np.dtype("<f4"), path.name, f"x-axis for spectrum {index + 1}"
                )
                if index == 0:
                    shared_x = spectrum_x
                elif spectrum_x.shape != shared_x.shape or not np.allclose(spectrum_x, shared_x, rtol=1e-7, atol=1e-9):
                    raise DataValidationError(f"{path.name}: SPC sub-spectra use different x-axes")

            exponent = sub_exponent if multi_exponent else global_exponent
            if exponent == 0x80:
                y, offset = _read_spc_values(
                    content, offset, count, np.dtype("<f4"), path.name, f"spectrum {index + 1}"
                )
            else:
                dtype = np.dtype("<i2") if precision_16 else np.dtype("<i4")
                raw_y, offset = _read_spc_values(
                    content, offset, count, dtype, path.name, f"spectrum {index + 1}"
                )
                bits = 16 if precision_16 else 32
                y = raw_y * float(2.0 ** (exponent - bits))
            spectra.append(y)
            label_index = sub_index if spectrum_count > 1 else index + 1
            labels.append(
                f"{path.name}[{sub_time:g} s]" if np.isfinite(sub_time) else f"{path.name}[{label_index}]"
            )

        header = [comment] if comment else []
        data_type = "raman" if x_type == 13 else "generic"
        x_label = "Raman shift / cm⁻¹" if data_type == "raman" else ""
        return LoadedFile(
            path=path,
            x=shared_x,
            ys=np.vstack(spectra),
            header=header,
            labels=labels,
            detected_delimiter="native binary SPC",
            data_type=data_type,
            x_label=x_label,
        )
    except DataValidationError:
        raise
    except Exception as exc:
        raise DataValidationError(f"{path.name}: could not read SPC data: {exc}") from exc


def _read_spc_values(
    content: bytes,
    offset: int,
    count: int,
    dtype: np.dtype,
    filename: str,
    description: str,
) -> tuple[np.ndarray, int]:
    byte_count = count * dtype.itemsize
    end = offset + byte_count
    if count < 1 or end > len(content):
        raise DataValidationError(f"{filename}: truncated or invalid {description}")
    values = np.frombuffer(content, dtype=dtype, count=count, offset=offset).astype(np.float64)
    return values, end


def _detect_text_axis(path: Path, header_lines: list[str], x: np.ndarray) -> tuple[str, str]:
    """Infer only common Raman/PXRD axes; uncertain data remains generic."""
    description = "\n".join([path.name, *header_lines]).casefold()
    normalized = re.sub(r"[\s_\-]+", " ", description)
    if path.suffix.lower() == ".raman" or "raman" in normalized:
        return "raman", "Raman shift / cm⁻¹"

    two_theta_tokens = ("2theta", "2 theta", "two theta", "2θ")
    pxrd_tokens = ("pxrd", "powder diffraction", "x ray diffraction", "xrd", "scattering vector")
    if any(token in normalized for token in two_theta_tokens):
        return "pxrd", "2θ / °"
    if any(token in normalized for token in pxrd_tokens) or path.suffix.lower() == ".chi":
        return "pxrd", "q / nm⁻¹"
    if re.search(r"(?:^|[,;\t ])q\s*(?:/|\[|\()", normalized):
        label = "q / Å⁻¹" if "angstrom" in normalized or "å" in normalized else "q / nm⁻¹"
        return "pxrd", label

    finite_x = np.asarray(x, dtype=np.float64)
    if finite_x.size and np.isfinite(finite_x).all():
        low = float(np.min(finite_x)); high = float(np.max(finite_x))
        if high - low >= 200.0 and high > 200.0:
            return "raman", "Raman shift / cm⁻¹"
        if low >= 0.0 and high <= 180.0:
            return "pxrd", "q / nm⁻¹"
    return "generic", "x"


def _extract_column_labels(
    header_lines: list[str],
    delimiter_name: str,
    expected_columns: int,
    filename: str,
) -> list[str]:
    delimiter = _DELIMITERS[delimiter_name]
    candidates: list[tuple[int, list[str]]] = []
    for raw_line in header_lines:
        if delimiter is None:
            parts = raw_line.split()
        else:
            try:
                parts = next(csv.reader([raw_line], delimiter=delimiter))
            except csv.Error:
                continue
            while parts and not parts[-1].strip():
                parts.pop()
        if len(parts) != expected_columns:
            continue
        labels: list[str] = []
        for index, value in enumerate(parts[1:], start=1):
            label = value.strip()
            labels.append(label or f"{filename}[{index}]")
        meaningful = [label.casefold() for label in labels if label and not label.startswith(f"{filename}[")]
        candidates.append((len(set(meaningful)), labels))
    return max(candidates, key=lambda candidate: candidate[0])[1] if candidates else []


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DataValidationError(f"{path.name}: text encoding could not be detected")


def _detect_data_start(lines: list[str], requested_delimiter: str) -> tuple[int, str]:
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        candidates = _candidate_delimiters(requested_delimiter)
        for name in candidates:
            values = _parse_numeric_line(line.strip(), _DELIMITERS[name])
            if values is not None and len(values) >= 2:
                return index, name
    raise DataValidationError("No numeric data rows were found")


def _detect_delimiter(lines: list[str], requested_delimiter: str) -> str:
    if requested_delimiter != "auto":
        if requested_delimiter not in _DELIMITERS:
            raise DataValidationError(f"Unknown delimiter setting: {requested_delimiter}")
        return requested_delimiter
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        for name in _candidate_delimiters("auto"):
            values = _parse_numeric_line(line.strip(), _DELIMITERS[name])
            if values is not None and len(values) >= 2:
                return name
    raise DataValidationError("Could not detect a numeric data delimiter")


def _candidate_delimiters(requested: str) -> list[str]:
    if requested != "auto":
        return [requested]
    return ["tab", "semicolon", "comma", "whitespace"]


def _parse_numeric_line(line: str, delimiter: str | None) -> list[float] | None:
    parts = line.split(delimiter) if delimiter is not None else line.split()
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) < 2:
        return None
    try:
        values = []
        for part in parts:
            normalized = part.replace("D", "E").replace("d", "e")
            if delimiter == ";":
                normalized = normalized.replace(",", ".")
            values.append(float(normalized))
        return values
    except ValueError:
        return None
