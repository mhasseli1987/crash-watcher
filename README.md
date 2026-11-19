# Crash Watcher

**Final desktop application:** See [راهنمای-فارسی.md](راهنمای-فارسی.md) for installation,
calibration, monitoring, and Windows build instructions. Run `desktop_app.py`
from the project virtual environment, or use `build-windows.ps1` for the onedir
Windows distribution.

Python 3.11+ YELLOW HUNTER simulation engine with local screen recognition
and a Windows desktop interface. Auto-click is off by default: a confirmed round at or above
the click threshold skips `autoclick_skip_rounds` rounds (default 7) and then clicks the next
`autoclick_max_clicks` rounds (default 3 — the 8th, 9th and 10th after the yellow) whatever
they close at. A second high round inside that window neither restarts the wait nor extends
the burst; the process resets after the burst. Each click waits `autoclick_delay_ms`
(default 1000 ms) after the round closes so the site registers it, and that wait is hard-capped
at one second — no click is ever issued later than that. Both label modes click — the mode button only marks the log entry.

**Smart click (opt-in).** With `smart_click_enabled` the clicker first forecasts the next
dot color from the recent color mix and the color that used to follow the previous one
(`desktop/forecast.py`); it only clicks when that forecast reaches
`prediction_min_confidence` (default 0.60) and enough rounds have been seen, otherwise the
round is skipped. The wait-then-burst run after a high round still fires unconditionally.
Every round is appended to `logs/rounds.jsonl` (time, round, multiplier, color, decision,
forecast, clicks left), and every `review_window_s` (default 1800 s = 30 minutes) a review
is written to `logs/reviews.csv` plus a plain-text `logs/summary.txt` (rounds, clicks, skips,
forecast hit rate, model accuracies, and which model the bot keeps). The review is also shown
in the calibration tab, so the forecast is judged by measurement instead of belief. Every
click is attributed to the rule that asked for it, scored against the colour that really
followed, and the totals are written to `logs/session-report.txt` when monitoring stops —
that file is the one to hand back for analysis. The game draws independent rounds, so
expect the model to sit near the base rates (red ≈ 50%, green ≈ 40%, yellow ≈ 10%) most of
the time.

**Smart review and smart play (opt-in).** The «بررسی هوشمند ضریب‌ها» tab asks a local
OpenAI-compatible server (FreeLLMAPI, default `http://127.0.0.1:31415/v1`, dashboard key
required) about the four regions the user marked: multiplier history, the current
multiplier, the colour-dot chart and the bet-button row. The model answers one JSON object
(`bet`, optional `second`, `confidence`, `reason`); the answer is advice shown in the tab and
logged to `logs/smart-review.jsonl`. With `smart_play_enabled` the answer becomes a bet:
`smart_play_learning_minutes` (default 30) of learning counted from monitoring start, then
only hands whose confidence reaches `smart_play_min_confidence` (default 0.40) — and whose
measured accuracy is not lower — are bet; `smart_play_two_colors` lets the model click two
buttons in one hand (payouts ≥ 2x make the worst case break-even). The decision for the next
hand is prepared in a background thread while the current hand is still running, so the
six-second window between two multipliers only needs the click; answers more than three
rounds old are discarded. The existing rules keep priority over the model.

**Chart signal (on by default).** When two consecutive rounds are both yellow (two yellow
dots stacked in one column) the clicker arms `signal_repeat` (default 2) follow-up clicks:
every yellow round in that window is clicked immediately to bet on the round after it.
What came before the pair does not matter — only that two yellows landed back to back. The arm rises only when the shape appears
after being absent, so the same chart cannot keep re-arming itself. Scans are read from
`chart_region` (set it with «انتخاب شبکهٔ رنگ»); while the signal is armed it takes
precedence over the wait-then-burst rule.

A second chart shape does the same scan: a single column reading **yellow, green, green,
yellow** from the top is read as "more yellows are on the way", and the clicker then bets
yellow on each of the next `signal_burst_rounds` rounds (default 9). Like the first signal
it only arms on a rising edge and it does not click the round that shows the shape.

**Numbered regions.** The calibration tab labels the four marked areas of the screen as
region 1 (history strip), 2 (current multiplier), 3 (the Trends colour grid) and 4 (the row of
Bet Red / Bet Green / Bet Moon buttons). Region 4 is split into three equal parts, so one
selection is enough for all three bets. See [docs/regions.md](docs/regions.md).

**Session timer (off by default).** `session_minutes` defaults to 0, so monitoring runs
until it is stopped by hand and then writes the report. Setting it to a number ends the
session after that many minutes. The dashboard has the countdown: a badge next to the title
shows `HH:MM:SS` left (it turns red under five minutes), a progress bar shows how much of the
session has passed, and the presets *30 min / 1 h / 3 h / no timer* set the limit without
opening the calibration tab. The timer measures wall-clock time, so it keeps running while
monitoring is paused. `shutdown_after_session` is off by default; when it is on,
`shutdown /s /t 60` runs on Windows, giving a minute to abort with `shutdown /a`, and the
machine stays on otherwise.

**Special multipliers (on by default).** `value_bets` maps an exact closing multiplier to the
colour to bet on the next round — defaults are `1.00`, `1.11` and `1.99` → green and `2.08` →
yellow. Edit it in the calibration tab as `1.00:GREEN, 2.08:YELLOW`; matching is on the exact
`Decimal` value and it outranks the smart gate and the pair rule (a yellow run still wins,
because that run belongs to the yellow rule).

**Streak run (on by default).** A colour that closes more than `streak_signal_length` times
in a row (default 6, so seven rounds) is read as a trend that keeps going: that same colour
is bet on each of the next `streak_signal_clicks` rounds (default 10), whether the streak
survives or not. Both red and green streaks count; yellow is left to the yellow rules.

**Row signal (on by default).** The newest column fills top to bottom, so the next round's
dot always lands in the next free row of that column. When that same row was yellow exactly
`row_signal_gap` columns back (default 4), the repetition the user watches for is due on
that round and the bet is placed on that round itself — never on the round after the repeat
has already been drawn. Only the first `row_signal_rows` rows (4) count: below row four the
user rates the odds too low.

**Pair flip (on by default).** Two reds or two greens in a row (vertical neighbours) arm one
bet on the opposite colour: after two greens the next round clicks the red button, after two
reds the green one. Capture the three bet buttons once with «دکمهٔ قرمز / سبز / زرد»
(`bet_point_red`, `bet_point_green`, `bet_point_yellow`); without them the normal click
target is used. A smart click now also targets the colour the model predicted.

## Windows setup and execution

Run in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m examples.strategy_simulation
```

## API

```python
from crash_watcher import StrategyConfig, StrategyEngine

engine = StrategyEngine(StrategyConfig(fixed_bet_amount=10))
trigger = engine.process_round("round_001", 12.45)  # no bet
result = engine.process_round("round_002", 10)  # win on attempt 1
print(result.bet_outcome, result.events)
```

A confirmed yellow starts a cycle. Each subsequent round places one simulated
fixed bet, terminating at the first yellow or after ten losses by default.
A winning yellow never triggers another cycle. Red is below 2; green is at least
2 and below 10; yellow is at least 10. Custom thresholds are supported.

Configuration, results, and events are immutable dataclasses; numeric amounts
and multipliers are normalized to Decimal. Inputs must be finite positive
numbers (int, float, Decimal); booleans, strings, zero, and negatives are rejected.
Round IDs are nonempty strings, normalized by trimming surrounding whitespace.
Invalid submissions raise ValueError without consuming their IDs or attempts.
Duplicate valid submissions return duplicate=True and only an ignored event;
they never register a bet or change cycle state, even if their payload differs.
Input validation precedes duplicate detection.

RoundResult contains the ID, multiplier, category, states before/after, trigger
and bet flags, amount, executed attempt, remaining attempts, outcomes, timestamp,
cycle ID, duplicate flag, and ordered events. Terminal results retain the executed
attempt (1–10), while engine counters and result remaining_attempts reset to zero.
Waiting states have zero remaining attempts. Use engine.attempt_number for the
post-transition counter. Timestamps are UTC by default; inject clock=callable
for reproducible results. The same timestamp is shared by all events in a result.

engine.reset() cancels the active cycle and returns a reset event. It preserves
processed IDs and cycle numbering so old rounds cannot be replayed. Create a new
engine for an independent session.

## Structure

- crash_watcher/core/config.py: validated immutable settings
- crash_watcher/core/enums.py: states, categories, outcomes, event types
- crash_watcher/core/models.py: immutable round results
- crash_watcher/core/events.py: structured events for future consumers
- crash_watcher/core/strategy_engine.py: state transitions and duplicate protection
- tests/: rule, edge case, and configuration tests
- examples/strategy_simulation.py: deterministic win, expiration, restart sequence
- crash_watcher/vision/: local capture, OCR, round detection, pipeline, events
- examples/visual_diagnostic.py: monitor calibration, screenshots and live diagnostics
- examples/vision_benchmark.py: local synthetic OCR benchmark

## Limits

Call serially from one owning thread. State and duplicate history are in memory,
with no persistence or bounded retention; duplicate history grows with input.
The caller supplies confirmed rounds in chronological order; the engine cannot
verify external round authenticity or ordering. This phase simulates outcomes
only and does not calculate balances, profit, or predict future results.

## Phase 2: local visual monitoring

Install requirements in the project virtual environment. Live recognition uses
the bundled RapidOCR model by default. A local Tesseract executable is an
optional alternative, selected with `--tesseract PATH`. No network or website
data is used by the running application.

Open the game manually in Firefox. First list monitors:

```powershell
.\.venv\Scripts\python.exe -m examples.visual_diagnostic --monitors
```

Choose a tight rectangle around **only the final multiplier text**, in desktop
pixel coordinates. Inspect a sample before monitoring:

```powershell
.\.venv\Scripts\python.exe -m examples.visual_diagnostic --monitor 1 --x 500 --y 300 --width 240 --height 80 --save
```

The screenshot is saved in the local `dataset/` directory. Coordinates above
are examples; calibrate for the actual monitor. To run the worker and display
the captured region in a development preview:

```powershell
.\.venv\Scripts\python.exe -m examples.visual_diagnostic --monitor 1 --x 500 --y 300 --width 240 --height 80 --interval-ms 100 --preview
```

Remove `--preview` for console-only monitoring. Press `q` in preview or Ctrl+C
to stop. The console prints recognition, confidence, color sample, detector
state, strategy events, and average capture/processing/OCR/detection timings.
Use `--tesseract "C:\path\to\tesseract.exe"` only if that executable is
installed. The optional diagnostic preview uses Tkinter and Pillow.

The detector ignores a result already visible at startup. It requires observed
motion, a stable blank or uncertain text phase, then repeated stable, valid OCR
before emitting a completed round. Stable animation values and a permanently
visible old result are therefore held back. It generates a UUID round ID for
each confirmed completion and submits it to the original `process_round()`
API exactly once. Uncertain readings never reach the strategy.
Capture failure or monitor geometry change cancels an active strategy cycle,
because unseen rounds could otherwise corrupt its attempt count.

Recognition accepts strict two-decimal values such as `10.00x`; malformed OCR,
ambiguous variants, and low-confidence readings are rejected. Color sampling
is diagnostic only. The monitor must show the multiplier in this format. The
detector is deliberately conservative: it can miss a valid result when no
visible blank phase separates rounds. It cannot prove that a game has ended
from generic pixels alone. Screen occlusion, resizing, UI changes and OCR
errors require recalibration. Actual Firefox screenshots are still needed to
measure real OCR accuracy, tune color/thresholds and verify final-result cues.

Synthetic recognition and state-machine cases are covered by `tests/test_vision.py`.
Use `python -m examples.vision_benchmark` for local capture and real bundled
OCR timings on synthetic multiplier images. Production screenshots and true
round detection latency have not yet been benchmarked.
