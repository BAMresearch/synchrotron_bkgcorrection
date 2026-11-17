# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

project_root = Path(SPECPATH)

# Conda keeps the native PySide/Shiboken/Qt runtime in ``Library/bin`` instead
# of next to the Python extension modules.  PyInstaller otherwise finds the
# .pyd files but cannot resolve their DLL imports, which only works on the
# build PC while Conda's directory happens to be on PATH.
conda_bin = Path(sys.prefix) / "Library" / "bin"
qt_runtime_binaries = []
if conda_bin.is_dir():
    os.environ["PATH"] = str(conda_bin) + os.pathsep + os.environ.get("PATH", "")
    for dll_name in (
        "shiboken6.cp310-win_amd64.dll",
        "pyside6.cp310-win_amd64.dll",
        "Qt6Core.dll",
        "Qt6Gui.dll",
        "Qt6Widgets.dll",
        "Qt6Network.dll",
        "Qt6Svg.dll",
    ):
        dll_path = conda_bin / dll_name
        if dll_path.exists():
            qt_runtime_binaries.append((str(dll_path), "."))

a = Analysis(
    [str(project_root / "background_correction_studio.py")],
    pathex=[str(project_root)],
    binaries=qt_runtime_binaries,
    datas=[
        (str(project_root / "background_app" / "assets" / "app_icon.svg"), "assets"),
        (str(project_root / "background_app" / "assets" / "app_icon.png"), "assets"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "pandas", "pyspectra", "spc_spectra", "spc", "tkinter"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BackgroundCorrectionStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(project_root / "background_app" / "assets" / "app_icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BackgroundCorrectionStudio",
)
