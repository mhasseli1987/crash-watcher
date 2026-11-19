"""Launch the Windows desktop app without starting monitoring."""
import sys
from PySide6.QtWidgets import QApplication, QMessageBox
from .ui import MainWindow

def main() -> int:
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
    except Exception as exc:
        QMessageBox.critical(None, "Crash Watcher", str(exc))
        return 1
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
