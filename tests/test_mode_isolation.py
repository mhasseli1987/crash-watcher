"""Classic and Turnbal retain separate controls and decision formats."""
from dataclasses import replace
from unittest.mock import patch

from crash_watcher.desktop.config import DesktopSettings


def test_mode_switch_enables_only_turnbal_controls(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from crash_watcher.desktop import ui

    app = QApplication.instance() or QApplication([])
    with patch.object(ui, "load_settings", return_value=DesktopSettings(bot_mode="classic")):
        window = ui.MainWindow()
        try:
            controls = (window.threshold, window.click_wait, window.pair_flip,
                        window.smart_two, window.smart_rules)
            assert not window.tabs.isTabEnabled(3)
            assert all(not control.isEnabled() for control in controls)
            window.settings = replace(window.settings, bot_mode="turnbal")
            window._refresh_bot_mode()
            assert window.tabs.isTabEnabled(3)
            assert all(control.isEnabled() for control in controls)
        finally:
            window.close()
            app.processEvents()


def test_header_mode_button_switches_and_explains_active_monitor(monkeypatch):
    from unittest.mock import Mock
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from crash_watcher.desktop import ui
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, "load_settings", lambda: DesktopSettings())
    monkeypatch.setattr(ui, "save_settings", lambda settings: None)
    notice = Mock()
    monkeypatch.setattr(ui.QMessageBox, "information", notice)
    window = ui.MainWindow()
    window.mode_chip.click()
    assert window.settings.bot_mode == "turnbal" and window.tabs.isTabEnabled(3)
    window.mode_chip.click()
    assert window.settings.bot_mode == "classic"
    window.worker = Mock()
    window.worker.isRunning.return_value = True
    window.mode_chip.click()
    assert window.settings.bot_mode == "classic" and notice.called
    window.worker = None
    window.close()
    app.processEvents()
