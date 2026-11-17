from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6 import QtCore, QtGui, QtSvg


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    asset_dir = root / "background_app" / "assets"
    svg_path = asset_dir / "app_icon.svg"
    png_path = asset_dir / "app_icon.png"
    ico_path = asset_dir / "app_icon.ico"

    application = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    renderer = QtSvg.QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise RuntimeError(f"Could not read {svg_path}")
    image = QtGui.QImage(512, 512, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    renderer.render(painter)
    painter.end()
    if not image.save(str(png_path), "PNG"):
        raise RuntimeError(f"Could not write {png_path}")

    with Image.open(png_path) as source:
        source.save(ico_path, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"Generated {png_path.name} and {ico_path.name}")
    application.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
