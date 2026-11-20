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
                                                clamp_exit, default_exit, exit_bounds,
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


def test_prompt_carries_no_fixed_betting_rules():
    """در AI Analysis مدل باید خودش تحلیل کند، نه اینکه قواعد ربات را اجرا کند."""
    snapshot = SmartSnapshot(recent=("1.66", "4.18"), current="8.01",
                             grid_columns=(("Y", "G"),), rounds_seen=3)
    prompt = build_prompt(snapshot, threshold="10.00")
    for old_rule in ("هفت دست صبر", "قواعد کاربر", "ضرایب 1.00", "جفت سبز یا جفت قرمز",
                     "همان رنگ برای ده دست"):
        assert old_rule not in prompt
    for data in ("1. 1.66 (قرمز)", "4.18 (سبز)", "ستون 1", "3"):
        assert data in prompt


def test_prompt_asks_for_a_cashout_exit_in_classic_mode():
    snapshot = SmartSnapshot(recent=("1.66",), current="1.20")
    prompt = build_prompt(snapshot, classic=True)
    for expected in ('"exit"', "1.30", "1.85", "1.90", "1.10"):
        assert expected in prompt
    assert '"exit"' not in build_prompt(snapshot)


def test_parse_verdict_reads_the_cashout_exit_inside_the_range():
    verdict = parse_verdict('{"bet":"GREEN","exit":1.85,"confidence":0.6}', "1.20")
    assert verdict is not None and verdict.exit_at == 1.85
    # مدل هر عددی بدهد، خروج داخل بازهٔ ۱٫۳۰..۱٫۸۵ می‌ماند.
    assert parse_verdict('{"bet":"GREEN","exit":2.4}').exit_at == 1.85
    assert parse_verdict('{"bet":"GREEN","exit":1.05}').exit_at == 1.30
    # دست قبلی زیر ۱٫۱۰: تا ۱٫۹۰ مجاز است.
    low_last = parse_verdict('{"bet":"GREEN","exit":1.9}', "1.07")
    assert low_last is not None and low_last.exit_at == 1.90
    assert parse_verdict('{"bet":"GREEN","exit":1.9}', "1.20").exit_at == 1.85
    assert parse_verdict('{"bet":"GREEN","exit":"soon"}').exit_at is None


def test_exit_bounds_open_to_190_only_after_a_very_low_round():
    assert exit_bounds("1.09") == (1.30, 1.90)
    assert exit_bounds("1.00") == (1.30, 1.90)
    assert exit_bounds("1.10") == (1.30, 1.85)
    assert exit_bounds(None) == (1.30, 1.85)
    assert clamp_exit(1.95, "1.05") == 1.90
    assert default_exit("1.05") == 1.90
    assert default_exit("4.20") == 1.30


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


def test_ask_explains_a_timeout():
    def slow(request, timeout=None):
        raise TimeoutError("timed out")

    with mock.patch.object(smart_review.urlrequest, "urlopen", slow):
        with pytest.raises(RuntimeError) as error:
            ask("http://127.0.0.1:31415/v1", "", "auto", "سلام", timeout=90)
    assert "مهلت" in str(error.value)
    assert "90" in str(error.value)
    assert "timed out" not in str(error.value)


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
                {"smart_review_timeout_s": 30},
                {"smart_review_timeout_s": "45"},
                {"smart_review_every_rounds": 0},
                {"smart_review_enabled": "yes"}):
        with pytest.raises(ValueError):
            DesktopSettings(**bad).validate()

    path = tmp_path / "settings.json"
    settings = DesktopSettings(smart_review_enabled=True,
                               smart_review_url="http://127.0.0.1:4000/v1",
                               smart_review_key="sk-local", smart_review_model="glm-5.2",
                               smart_review_timeout_s=90, smart_review_every_rounds=2,
                               smart_review_rules="زرد بعد از سه سبز")
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded == settings
    assert load_settings(tmp_path / "absent.json").smart_review_url == \
        "http://127.0.0.1:31415/v1"


def test_old_timeout_default_is_raised_on_load(tmp_path: Path):
    path = tmp_path / "settings.json"
    for old in (30, 45, "90", 90.0, None):
        path.write_text(json.dumps({"smart_review_timeout_s": old}), encoding="utf-8")
        assert load_settings(path).smart_review_timeout_s == 120
    path.write_text(json.dumps({"smart_review_timeout_s": 200}), encoding="utf-8")
    assert load_settings(path).smart_review_timeout_s == 200


def test_migration_leaves_the_rest_of_an_old_file_alone(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"smart_review_timeout_s": 30, "smart_review_key": "sk-old",
                                "autoclick_threshold": "7.50",
                                "three_x_enabled": True,
                                "signal_rule_enabled": True}), encoding="utf-8")
    loaded = load_settings(path)
    assert loaded.smart_review_timeout_s == 120
    assert loaded.smart_review_key == "sk-old"
    assert loaded.autoclick_threshold == "7.50"
    # فایلی که موتور ۳x را دارد نباید قوانین قدیمی را خاموش کند.
    assert loaded.signal_rule_enabled is True
