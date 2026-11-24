import json
from urllib import error
from unittest import mock

import pytest

from crash_watcher.desktop.config import DesktopSettings, load_settings, save_settings
from crash_watcher.desktop import ollama_provider as ollama


class Reply:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.value).encode()


def test_defaults_and_legacy_settings(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"smart_review_enabled": true}', encoding="utf-8")
    settings = load_settings(path)
    assert not settings.ollama_enabled
    assert settings.ollama_model == "qwen3:1.7b"
    save_settings(settings, path)
    assert load_settings(path) == settings


@pytest.mark.parametrize("bad", [
    {"ollama_enabled": "yes"}, {"ollama_base_url": "https://example.com"},
    {"ollama_base_url": "http://127.0.0.1:11434/api"}, {"ollama_model": ""},
    {"ollama_timeout_ms": 0}, {"ollama_keep_alive": ""},
])
def test_bad_settings(bad):
    with pytest.raises(ValueError):
        DesktopSettings(**bad).validate()


def test_success_and_compact_request():
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append((req, timeout))
        if isinstance(req, str):
            return Reply({"models": [{"name": "qwen3:1.7b"}]})
        payload = json.loads(req.data)
        assert payload["format"]["properties"]["bet"]["enum"] == ["RED", "GREEN", "YELLOW", "SKIP"]
        assert payload["format"]["additionalProperties"] is False
        assert payload["keep_alive"] == "30s"
        assert payload["stream"] is False
        assert payload["think"] is False
        return Reply({"message": {"content": '{"bet":"GREEN","confidence":0.6}'}})

    with mock.patch.object(ollama.request, "urlopen", fake_urlopen):
        verdict = ollama.predict(DesktopSettings(ollama_enabled=True, bot_mode="turnbal"), "next round")
    assert verdict.bet == "GREEN"
    assert len(calls) == 2
    assert calls[1][1] == 4


def test_classic_schema_and_response_are_independent_of_colors():
    def fake(req, timeout=None):
        if isinstance(req, str):
            return Reply({"models": [{"name": "qwen3:1.7b"}]})
        payload = json.loads(req.data)
        assert payload["format"]["properties"]["bet"]["enum"] == ["PLAY", "SKIP"]
        return Reply({"message": {"content": '{"bet":"PLAY","confidence":0.6,"reason":"x"}'}})

    with mock.patch.object(ollama.request, "urlopen", fake):
        assert ollama.predict(DesktopSettings(ollama_enabled=True), "next round").bet == "PLAY"
    with pytest.raises(ollama.OllamaError):
        ollama._parse_verdict('{"bet":"RED","confidence":0.6}', ("PLAY", "SKIP"))


def test_missing_model_and_unavailable_service():
    settings = DesktopSettings(ollama_enabled=True)
    with mock.patch.object(ollama.request, "urlopen", return_value=Reply({"models": []})):
        with pytest.raises(ollama.OllamaError, match="not installed"):
            ollama.predict(settings, "prompt")
    with mock.patch.object(ollama.request, "urlopen", side_effect=error.URLError("refused")):
        with pytest.raises(ollama.OllamaError, match="unavailable"):
            ollama.predict(settings, "prompt")


@pytest.mark.parametrize("content", [
    "oops", '{"bet":"BLUE","confidence":0.5}',
    '{"bet":"RED","confidence":2}', '{"bet":"RED","confidence":true}',
    '{"bet":"RED","confidence":0.5,"second":"RED"}',
])
def test_malformed_predictions(content):
    def fake(req, timeout=None):
        return (Reply({"models": [{"name": "qwen3:1.7b"}]}) if isinstance(req, str)
                else Reply({"message": {"content": content}}))
    with mock.patch.object(ollama.request, "urlopen", fake):
        with pytest.raises(ollama.OllamaError):
            ollama.predict(DesktopSettings(ollama_enabled=True), "prompt")


def test_timeout():
    def fake(req, timeout=None):
        if isinstance(req, str):
            return Reply({"models": [{"name": "qwen3:1.7b"}]})
        raise TimeoutError("timed out")
    with mock.patch.object(ollama.request, "urlopen", fake):
        with pytest.raises(ollama.OllamaError, match="timed out"):
            ollama.predict(DesktopSettings(ollama_enabled=True), "prompt")
