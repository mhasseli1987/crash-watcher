import json
from pathlib import Path
from urllib import error as urlerror
from unittest import mock

import pytest

from crash_watcher.desktop import smart_review
from crash_watcher.desktop.color_grid import Dot, GridReading
from crash_watcher.desktop.config import DesktopSettings, load_settings, save_settings
from crash_watcher.desktop.smart_review import (SmartSnapshot, SmartVerdict, ask,
                                                build_prompt, chat_completions_url,
                                                grid_columns, parse_verdict,
                                                probe_models, review)


class _Reply:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _reading(columns):
    dots = []
    for column, colours in enumerate(columns):
        for row, colour in enumerate(colours):
            dots.append(Dot(colour, column, row, column * 20, row * 20))
    return GridReading(tuple(dots), len(columns), max(len(c) for c in columns), 20.0)


def test_chat_url_accepts_host_v1_and_full_url():
    assert chat_completions_url("http://127.0.0.1:31415") == \
        "http://127.0.0.1:31415/v1/chat/completions"
    assert chat_completions_url("http://127.0.0.1:31415/v1/") == \
        "http://127.0.0.1:31415/v1/chat/completions"
    full = "http://localhost:31415/v1/chat/completions"
    assert chat_completions_url(full) == full
    with pytest.raises(ValueError):
        chat_completions_url("127.0.0.1:31415")
    with pytest.raises(ValueError):
        chat_completions_url("  ")


def test_grid_columns_reads_newest_column_first():
    reading = _reading([["YELLOW", "GREEN", "GREEN"], ["RED", "RED"]])
    assert grid_columns(reading) == (("Y", "G", "G"), ("R", "R"))
    assert grid_columns(None) == ()


def test_prompt_covers_the_four_marked_regions():
    snapshot = SmartSnapshot(recent=("1.66", "4.18", "8.01"), current="47.23",
                             grid_columns=(("Y", "G", "G"), ("R",)), rounds_seen=42)
    prompt = build_prompt(snapshot, threshold="10.00", extra_rules="زرد بعد از سه سبز")
    for expected in ("1.66", "4.18", "47.23", "ستون 1", "قرمز", "سبز", "زرد",
                     "42", "10.00", "JSON", "زرد بعد از سه سبز", '"second"', "40%"):
        assert expected in prompt
    single = build_prompt(snapshot, min_confidence=0.5, allow_two=False)
    assert "second" not in single and "50%" in single


def test_parse_verdict_accepts_plain_fenced_and_persian_answers():
    assert parse_verdict('{"bet":"GREEN","confidence":0.72,"reason":"روند سبز"}') == \
        SmartVerdict("GREEN", 0.72, "روند سبز")
    fenced = '```json\n{"bet":"yellow","confidence":"65%","reason":"x"}\n```'
    assert parse_verdict(fenced) == SmartVerdict("YELLOW", 0.65, "x")
    assert parse_verdict('{"bet":"مون","confidence":80}').bet == "YELLOW"
    assert parse_verdict('{"bet":"skip"}').bet == "SKIP"
    assert parse_verdict('{"bet":"BLUE"}') is None
    assert parse_verdict("سبز") is None
    assert parse_verdict("") is None
    # مدل گاهی قبل از JSON توضیح می‌نویسد و در توضیح آکولاد دارد
    talkative = ('Let me analyze {this} carefully.\n'
                 '{"bet":"GREEN","confidence":0.55,"reason":"جفت قرمز"}')
    assert parse_verdict(talkative) == SmartVerdict("GREEN", 0.55, "جفت قرمز")


def test_ask_posts_an_openai_chat_payload():
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Reply({"choices": [{"message": {"content": '{"bet":"RED"}'}}]})

    with mock.patch.object(smart_review.urlrequest, "urlopen", fake_urlopen):
        answer = ask("http://127.0.0.1:31415/v1", "sk-test", "glm-5.2", "سلام", timeout=12)
    assert answer == '{"bet":"RED"}'
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["model"] == "glm-5.2"
    assert captured["body"]["messages"][-1]["content"] == "سلام"
    assert captured["body"]["stream"] is False
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["timeout"] == 12


def test_ask_explains_an_unreachable_port():
    def refuse(request, timeout=None):
        raise urlerror.URLError(ConnectionRefusedError("refused"))

    with mock.patch.object(smart_review.urlrequest, "urlopen", refuse):
        with pytest.raises(RuntimeError) as error:
            ask("http://127.0.0.1:31415/v1", "", "auto", "سلام")
    assert "31415" in str(error.value)
    assert "FreeLLMAPI" in str(error.value)


def test_ask_reports_a_rejected_key():
    def unauthorised(request, timeout=None):
        raise urlerror.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    with mock.patch.object(smart_review.urlrequest, "urlopen", unauthorised):
        with pytest.raises(RuntimeError) as error:
            ask("http://127.0.0.1:31415/v1", "bad", "auto", "سلام")
    assert "کلید" in str(error.value)


def test_probe_models_returns_ids():
    with mock.patch.object(smart_review.urlrequest, "urlopen",
                           lambda request, timeout=None: _Reply({"data": [{"id": "auto"},
                                                                          {"id": "glm-5.2"}]})):
        assert probe_models("http://127.0.0.1:31415/v1", "k") == ["auto", "glm-5.2"]


def test_review_returns_verdict_raw_and_prompt():
    seen = {}

    def fake_ask(base_url, key, model, prompt, timeout=45.0, system=None):
        seen["prompt"] = prompt
        return '{"bet":"GREEN","confidence":0.6,"reason":"جفت سبز"}'

    with mock.patch.object(smart_review, "ask", fake_ask):
        verdict, raw, prompt = review("http://127.0.0.1:31415/v1", "", "auto",
                                      SmartSnapshot(recent=("2.10",), current="2.10"))
    assert verdict == SmartVerdict("GREEN", 0.6, "جفت سبز")
    assert raw.startswith('{"bet"')
    assert prompt == seen["prompt"]


def test_smart_review_settings_are_validated_and_persisted(tmp_path: Path):
    DesktopSettings(smart_review_url="http://127.0.0.1:31415/v1").validate()
    for bad in ({"smart_review_url": "127.0.0.1:31415"},
                {"smart_review_timeout_s": 4},
                {"smart_review_timeout_s": "45"},
                {"smart_review_every_rounds": 0},
                {"smart_review_enabled": "yes"}):
        with pytest.raises(ValueError):
            DesktopSettings(**bad).validate()

    path = tmp_path / "settings.json"
    settings = DesktopSettings(smart_review_enabled=True,
                               smart_review_url="http://127.0.0.1:4000/v1",
                               smart_review_key="sk-local", smart_review_model="glm-5.2",
                               smart_review_timeout_s=30, smart_review_every_rounds=2,
                               smart_review_rules="زرد بعد از سه سبز")
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded == settings
    assert load_settings(tmp_path / "absent.json").smart_review_url == \
        "http://127.0.0.1:31415/v1"


def test_verdict_cashout_is_clamped_to_the_140_190_range():
    assert parse_verdict('{"bet":"RED","cashout":1.20}').cashout == 1.40
    assert parse_verdict('{"bet":"RED","cashout":2.50}').cashout == 1.90
    assert parse_verdict('{"bet":"RED","cashout":1.65}').cashout == 1.65
    assert parse_verdict('{"bet":"RED"}').cashout is None
