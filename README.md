# Crash Watcher

**Final desktop application:** See [راهنمای-فارسی.md](راهنمای-فارسی.md) for installation,
calibration, monitoring, and Windows build instructions. Run `desktop_app.py`
from the project virtual environment, or use `build-windows.ps1` for the onedir
Windows distribution.

Python 3.11+ YELLOW HUNTER simulation engine with local screen recognition
and a Windows desktop interface. It does not click betting controls or place wagers.

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
