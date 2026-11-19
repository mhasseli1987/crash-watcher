"""Entry point. DPI awareness must be set before Qt is imported."""
from __future__ import annotations

import ctypes
import sys


def _dpi() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


_dpi()

from PySide6.QtWidgets import QApplication, QMessageBox


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CrashWatcher")
    try:
        from .ui import MainWindow
        window = MainWindow()
    except Exception as exc:
        QMessageBox.critical(None, "CrashWatcher", f"راه‌اندازی ناموفق:\n{exc}")
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())