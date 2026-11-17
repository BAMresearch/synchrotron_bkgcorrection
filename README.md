# Background Correction Studio

Background Correction Studio is a self-contained desktop workflow for importing spectral data series, subtracting smooth backgrounds, normalizing and integrating signals, previewing results, and exporting publication-ready plots.

It modernizes the legacy `Controller.py` workflow while keeping that code untouched for numerical comparison. The application starts without `pyspectra` or Pandas. Ordinary XY-family files and modern little-endian SPC files are parsed directly.

## Highlights

- Drag-and-drop file list with validation, row reordering, enable/disable controls, and raw previews
- Direct import of `.xy`, `.dat`, `.chi`, `.txt`, `.raman`, `.csv`, `.tsv`, and `.spc`
- Native multi-spectrum SPC reader with acquisition-time labels and no obsolete SPC dependency
- Automatic Raman/PXRD recognition with a visible manual data-type override
- Raman-aware controls that preserve Raman shift and hide PXRD-only q/2-theta/wavelength conversion
- Editable scientific x-label dropdown with Raman shift, 2-theta, q, and generic choices
- Strict x-axis validation or explicit interpolation onto the first file's axis
- Asymmetric Least Squares (ALS) and robust arPLS background subtraction
- Optional jar/reference subtraction with a selectable scaling range and drag-and-drop selection of one embedded SPC/CSV scan
- Grounding and area, sum, or maximum normalization
- ROI area/maximum extraction and normalization
- Two-click ROI selection and shaded ROI ranges directly in the curve preview
- Live curve previews and final heatmap preview
- Permanently visible, debounced live baseline preview while tuning correction parameters
- Automatic `Time / min` heatmap axis from a configurable seconds-per-scan interval, with a scan-number fallback
- Full control over the x label, colormap, robust/HDR/per-scan heatmap scaling, color limits, axis direction, output size, DPI, and image format
- Exact horizontal plot bounds without Matplotlib padding beyond the first and last x values
- Adaptive multicore processing with a safe single-worker fallback on restricted PCs
- Responsive processing, progress, ETA, cancellation, and an activity log
- Saveable `.bcproject` projects and JSON settings presets
- UTF-8 data export and a complete JSON run manifest for reproducibility

## Run from source

The development environment requires Python 3.10 or later plus the packages in `requirements.txt`.

```powershell
python background_correction_studio.py
```

If a legacy `settings.json` is present on startup, the app migrates its settings in memory. Otherwise it uses built-in defaults.

In the curve preview, tick **Separate reference and baseline** to show the scaled jar/reference contribution and the fitted residual baseline as separate lines. Drag data rows to reorder the series. To use an imported scan as a reference, open **Reference** and drop its row onto **Reference source**, or choose **Use selected item**.

## Typical workflow

1. Add files or use **Add data folder…** to import all supported data files from a selected folder. Each row shows curve count, point count, and x-range; errors are shown in red with details in the tooltip and activity log.
2. Drag rows, use Move up/down, or disable files as needed. Curves are processed in the displayed order.
3. Review import, range, unit, background, and normalization settings under **Processing**. Select CPU workers from **Settings → CPU workers**.
4. Configure optional reference subtraction and ROIs.
5. Select plot and output options. The heatmap uses elapsed minutes by default; enter the acquisition time represented by one scan under **Time per scan**.
6. Click **Run correction**, choose the destination folder when prompted, and start the run. The app stays responsive and displays progress and an ETA.
7. Inspect processed curves or the result heatmap in **Preview**.

The **Output** tab controls which files are exported; it no longer silently creates a results folder beside the source data. A run can produce:

- `<name>_corrected.dat`
- `<name>_baseline.dat`
- optional original and reference-corrected matrices
- `<name>_rois.csv`
- `<name>.png`, `.pdf`, `.svg`, or `.tiff`
- `<name>_run.json` containing sources, timing, worker count, and every processing setting

## Performance behavior

Automatic mode deliberately avoids multiprocessing for small jobs because Windows process startup can cost more than correcting a few short curves. Larger datasets are processed in batches using a persistent process pool. If process creation is blocked by corporate security policy, the run continues with one worker and records the fallback in the activity log.

To benchmark a representative synthetic workload:

```powershell
python scripts\benchmark.py --curves 100 --points 500 --workers 1
python scripts\benchmark.py --curves 100 --points 500 --workers 0
```

## Tests

The test suite covers legacy ALS compatibility, arPLS stability, settings migration, text parsing, final-row preservation, delimiter detection, x-axis interpolation, units, normalization, ROIs, export, and headless UI construction.

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -v
```

## Portable Windows build

Only the build PC needs Python and packages. Target PCs receive the generated `dist\BackgroundCorrectionStudio` folder and do not need Python, `pip`, internet access, or administrator rights.

The portable build deliberately excludes the obsolete SPC libraries. Its built-in reader supports modern little-endian Galactic SPC files directly, including multi-spectrum Raman series such as Optical Spectra Xplorer exports.

```powershell
python -m pip install -r requirements.txt
python -m pip install "pyinstaller>=6,<7"
powershell -ExecutionPolicy Bypass -File scripts\build_portable.ps1
```

Use `-SkipArchive` during rapid local test cycles to rebuild only the unpacked application folder.

The build script creates a one-folder distribution first because it starts faster and is easier to diagnose than a self-extracting single executable. It also creates `dist\BackgroundCorrectionStudio-portable.zip` for distribution.

## Architecture

- `background_app/io.py` — robust text and native binary SPC import
- `background_app/baseline.py` — cached sparse ALS and arPLS implementations
- `background_app/processing.py` — unit conversion, reference correction, normalization, adaptive batching, and ROIs
- `background_app/plotting.py` — curve and heatmap figures
- `background_app/exporting.py` — UTF-8 data, plots, ROI tables, and run manifests
- `background_app/config.py` — typed configuration and legacy migration
- `background_app/ui.py` — PySide6 desktop interface
- `background_correction_studio.py` — application entry point

The numerical core has no GUI or file-dialog dependencies, allowing deterministic testing and future use from a command-line or automated pipeline.

## Legacy compatibility

With the legacy ALS parameters and convergence disabled, the new implementation matches the original calculation to floating-point precision. The current regression test observes a maximum absolute difference on the order of `10⁻¹⁰` for its reference spectrum.

The original files remain in the repository for side-by-side validation. New development should target `background_app` and `background_correction_studio.py`.
