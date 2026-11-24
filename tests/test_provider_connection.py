"""Connection diagnostics only; no strategy or click execution."""
import json
from unittest import mock
from urllib.error import HTTPError, URLError

import pytest

from crash_watcher.desktop import smart_review


class Reply:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


@pytest.mark.parametrize("body", [
    b"<html>service unavailable</html>", b"[]", b"{}",
    b'{"error":"upstream unavailable"}', b'{"data":{}}',
    b'{"data":[null]}', b'{"data":[{}]}',
    b'{"data":[{"id":7}]}', b'{"data":[{"id":" "}]}',
])
def test_bad_models_response_is_not_reported_as_connected(body):
    with mock.patch.object(smart_review.urlrequest, "urlopen", return_value=Reply(body)):
        with pytest.raises(RuntimeError):
            smart_review.probe_models("https://provider.example/v1", "")


def test_valid_empty_models_list_is_accepted():
    with mock.patch.object(smart_review.urlrequest, "urlopen",
                           return_value=Reply(json.dumps({"data": []}).encode())):
        assert smart_review.probe_models("https://provider.example/v1", "") == []


def test_external_connection_error_does_not_suggest_starting_a_local_server():
    message = smart_review._error_message(URLError(TimeoutError("timed out")),
                                           "https://provider.example/v1/models")
    assert "FreeLLMAPI" not in message
    assert "خارجی" in message


def test_auth_error_does_not_assume_one_provider():
    message = smart_review._error_message(
        HTTPError("https://provider.example", 403, "Forbidden", {}, None),
        "https://provider.example")
    assert "403" in message
    assert "دسترسی" in message
    assert "FreeLLMAPI" not in message


def test_repeated_connection_test_preserves_running_thread(monkeypatch):
    from threading import Event

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QApplication
    from crash_watcher.desktop.config import DesktopSettings
    from crash_watcher.desktop import ui

    app = QApplication.instance() or QApplication([])
    release = Event()

    def slow_probe(*args):
        release.wait(3)
        return ["health-check-model"]

    with mock.patch.object(ui, "load_settings", return_value=DesktopSettings()), \
            mock.patch.object(ui, "probe_models", side_effect=slow_probe):
        window = ui.MainWindow()
        try:
            window.test_smart_connection()
            original = window.smart_probe
            window.test_smart_connection()
            assert window.smart_probe is original
            closing = QCloseEvent()
            window.closeEvent(closing)
            assert not closing.isAccepted()
        finally:
            release.set()
            assert window.smart_probe.wait(4000)
            app.processEvents()
            window.close()
        assert "health-check-model" in window.smart_probe_status.text()
