# Phase 3 verification — 2026-11-04

- Automated suite: 131 passed (`python -m pytest -q`).
- Bundled Windows application: starts with a responsive desktop window.
- Bundled OCR self-test: passed; read `Bang @13.05x` and wrote `self-test.json`.
- Synthetic final-result OCR: recognized `Bang @1.46x`, `Bang @7.50x`, and `Bang @13.05x`.
- Supplied live multiplier and countdown screenshots: correctly rejected as final results.
- Real finished Firefox round: not yet observed or calibrated against a supplied `Bang @...` screenshot. Live-game accuracy is therefore unverified.
- No real betting input is sent by the application.
