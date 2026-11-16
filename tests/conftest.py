"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def isolated_clicker_logs(tmp_path, monkeypatch):
    """Keep test writes out of the project's real logs/ folder.

    AutoClicker falls back to PROJECT_ROOT / "logs" for its round log, summary
    and session report, and MainWindow appends reviews.csv / events.csv there.
    Pointing each module's PROJECT_ROOT at the test's own tmp_path keeps the
    real logs/ folder meaningful for real sessions.
    """
    from crash_watcher.desktop import clicker

    monkeypatch.setattr(clicker, "PROJECT_ROOT", tmp_path)
    try:
        from crash_watcher.desktop import ui
    except Exception:  # Qt not installed: nothing to redirect
        return
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
