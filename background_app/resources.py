from __future__ import annotations

from pathlib import Path

import matplotlib
from PySide6 import QtGui, QtWidgets


def install_ui_font(application: QtWidgets.QApplication) -> str:
    """Install a bundled, Unicode-capable UI font.

    This keeps text rendering predictable on locked-down and clean Windows PCs and
    also makes headless visual tests independent of system font discovery.
    """
    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
    families = QtGui.QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
    family = families[0] if families else "Sans Serif"
    application.setFont(QtGui.QFont(family, 9))
    return family
