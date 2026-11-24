from dataclasses import replace
from unittest.mock import Mock

import pytest

from crash_watcher import StrategyEngine
from crash_watcher.desktop.config import DesktopSettings, Region, load_settings, save_settings
from crash_watcher.desktop.smart_review import SmartSnapshot, SmartVerdict
from crash_watcher.desktop import worker as module


@pytest.mark.parametrize("mode,bet", [("classic", "PLAY"), ("turnbal", "GREEN")])
def test_local_only_never_calls_remote_provider(mode, bet, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    settings = DesktopSettings(ai_provider="local", bot_mode=mode,
                               result_region=Region(1, 0, 0, 300, 100))
    worker = module.MonitorWorker(settings, StrategyEngine(settings.strategy()))
    remote = Mock(side_effect=AssertionError("remote request in local-only mode"))
    monkeypatch.setattr(module, "ask", remote)
    monkeypatch.setattr(module, "review_turnbal", remote)
    local = Mock(return_value=SmartVerdict(bet, 0.6, "local"))
    monkeypatch.setattr(module, "predict_ollama", local)
    reports = []
    worker.smart_report.connect(reports.append)
    version = worker._prediction.begin("next", 30)
    worker._hybrid_ai("next", version, SmartSnapshot(recent=("1.20", "2.50", "12.0")), settings)
    assert worker._prediction.freeze("next").source == "OLLAMA"
    assert not remote.called
    assert local.call_args.args[0].ollama_timeout_ms == settings.ollama_timeout_ms
    assert reports[-1]["status"] == "PREPARED" and reports[-1]["verdict"].bet == bet


def test_local_failure_is_visible_before_round_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    settings = DesktopSettings(ai_provider="local", result_region=Region(1, 0, 0, 300, 100))
    worker = module.MonitorWorker(settings, StrategyEngine(settings.strategy()))
    reports = []
    worker.smart_report.connect(reports.append)
    monkeypatch.setattr(module, "predict_ollama", Mock(side_effect=TimeoutError("slow model")))
    version = worker._prediction.begin("next", 30)
    worker._hybrid_ai("next", version, SmartSnapshot(), settings)
    assert reports[-1]["status"] == "ERROR" and "slow model" in reports[-1]["error"]


def test_no_prediction_is_requested_without_confirmed_history(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    settings = DesktopSettings(ai_provider="local", result_region=Region(1, 0, 0, 300, 100))
    worker = module.MonitorWorker(settings, StrategyEngine(settings.strategy()))
    launch = Mock(side_effect=AssertionError("request without confirmed data"))
    monkeypatch.setattr(module, "Thread", launch)
    worker._start_hybrid(Mock())
    assert not launch.called and worker._prediction_target is None


def test_local_provider_selection_persists(tmp_path):
    settings = DesktopSettings(ai_provider="local", ollama_model="qwen3:1.7b")
    path = tmp_path / "settings.json"
    save_settings(settings, path)
    assert load_settings(path) == settings


def test_discovery_prefers_installed_qwen(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from crash_watcher.desktop import ui
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, "load_settings", lambda: DesktopSettings())
    window = ui.MainWindow()
    window._local_models_found(["other:latest", "qwen3:8b"])
    assert window.local_model.currentText() == "qwen3:8b"
    discovery = Mock()
    monkeypatch.setattr(window, "discover_local_models", discovery)
    window.ai_provider.setCurrentIndex(window.ai_provider.findData("local"))
    assert window._settings_from_form().ai_provider == "local"
    assert discovery.called and not window.smart_url.isEnabled()
    window._local_models_found(["qwen3:8b"])
    assert "qwen3:8b" in window.smart_probe_status.text()
    window._set_local_status("local error")
    assert window.smart_probe_status.text() == window.local_status.text() == "local error"
    window.close()
    app.processEvents()
