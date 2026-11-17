from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

if "--smoke-test" in sys.argv:
    # Keep the verification path headless so it can be run on a clean build
    # machine without showing a window or inheriting the developer's Qt setup.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

def load_initial_settings(settings_type):
    settings_path = Path(__file__).resolve().parent / "settings.json"
    if settings_path.exists():
        try:
            return settings_type.load(settings_path)
        except Exception:
            pass
    return settings_type()


def apply_fixed_light_theme(application: QtWidgets.QApplication) -> None:
    """Use the same readable light palette regardless of the Windows theme."""
    application.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    palette = QtGui.QPalette()
    colors = {
        QtGui.QPalette.ColorRole.Window: "#f4f6f8",
        QtGui.QPalette.ColorRole.WindowText: "#20252b",
        QtGui.QPalette.ColorRole.Base: "#ffffff",
        QtGui.QPalette.ColorRole.AlternateBase: "#f7f9fb",
        QtGui.QPalette.ColorRole.ToolTipBase: "#fffbe6",
        QtGui.QPalette.ColorRole.ToolTipText: "#20252b",
        QtGui.QPalette.ColorRole.Text: "#20252b",
        QtGui.QPalette.ColorRole.Button: "#f0f2f4",
        QtGui.QPalette.ColorRole.ButtonText: "#20252b",
        QtGui.QPalette.ColorRole.BrightText: "#ffffff",
        QtGui.QPalette.ColorRole.Link: "#1769aa",
        QtGui.QPalette.ColorRole.LinkVisited: "#5c4b8a",
        QtGui.QPalette.ColorRole.Highlight: "#2f8bc2",
        QtGui.QPalette.ColorRole.HighlightedText: "#ffffff",
        QtGui.QPalette.ColorRole.PlaceholderText: "#6f7882",
    }
    for role, color in colors.items():
        palette.setColor(role, QtGui.QColor(color))
    for role in (
        QtGui.QPalette.ColorRole.WindowText,
        QtGui.QPalette.ColorRole.Text,
        QtGui.QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(
            QtGui.QPalette.ColorGroup.Disabled,
            role,
            QtGui.QColor("#747d86"),
        )
    palette.setColor(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.Base,
        QtGui.QColor("#edf0f2"),
    )
    application.setPalette(palette)


class StartupWindow(QtWidgets.QDialog):
    """Small staged progress window shown while the heavy modules are imported."""

    def __init__(self, icon: QtGui.QIcon):
        super().__init__()
        self.setWindowTitle("Starting Background Correction Studio")
        self.setWindowIcon(icon)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        self.setFixedSize(460, 142)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        title = QtWidgets.QLabel("Background Correction Studio")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #20252b;")
        self.message = QtWidgets.QLabel("Starting application…")
        self.message.setStyleSheet("color: #4f5b66;")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(2)
        self.progress.setTextVisible(True)
        self.timing = QtWidgets.QLabel("0.0 s elapsed")
        self.timing.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.timing.setStyleSheet("color: #6b747d; font-size: 11px;")
        layout.addWidget(title)
        layout.addWidget(self.message)
        layout.addWidget(self.progress)
        layout.addWidget(self.timing)
        self.setStyleSheet(
            "QDialog { background: #f4f6f8; color: #20252b; }"
            "QProgressBar { border: 1px solid #b8c0c8; border-radius: 4px; "
            "background: #ffffff; color: #20252b; text-align: center; min-height: 20px; }"
            "QProgressBar::chunk { background: #2f8bc2; border-radius: 3px; }"
        )
        self._elapsed = QtCore.QElapsedTimer()
        self._elapsed.start()

    def advance(self, value: int, message: str) -> None:
        self.progress.setValue(value)
        self.message.setText(message)
        elapsed = self._elapsed.elapsed() / 1000.0
        if 5 <= value < 100:
            eta = max(0.0, elapsed * (100 - value) / value)
            self.timing.setText(f"{elapsed:.1f} s elapsed · about {eta:.1f} s remaining")
        else:
            self.timing.setText(f"{elapsed:.1f} s elapsed")
        QtWidgets.QApplication.processEvents()


def main() -> int:
    multiprocessing.freeze_support()
    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName("Background Correction Studio")
    application.setOrganizationName("BAM")
    apply_fixed_light_theme(application)
    if hasattr(sys, "_MEIPASS"):
        icon_path = Path(sys._MEIPASS) / "assets" / "app_icon.png"
    else:
        icon_path = Path(__file__).resolve().parent / "background_app" / "assets" / "app_icon.png"
    icon = QtGui.QIcon(str(icon_path)) if icon_path.exists() else QtGui.QIcon()
    application.setWindowIcon(icon)

    startup = StartupWindow(icon)
    if "--smoke-test" not in sys.argv:
        startup.show()
        startup.advance(5, "Loading application settings…")

    from background_app.config import AppSettings

    startup.advance(18, "Loading plotting engine…")
    from background_app.resources import install_ui_font

    startup.advance(42, "Loading numerical processing engine…")
    from background_app.ui import MainWindow

    startup.advance(78, "Preparing user interface…")
    install_ui_font(application)
    settings = load_initial_settings(AppSettings)
    startup.advance(88, "Building controls and preview…")
    window = MainWindow(settings)
    startup.advance(98, "Opening application…")
    window.show()
    startup.advance(100, "Ready")
    startup.close()
    if "--smoke-test" in sys.argv:
        QtCore.QTimer.singleShot(750, application.quit)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
