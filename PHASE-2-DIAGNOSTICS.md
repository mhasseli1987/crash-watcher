# Phase 2 diagnostic report

Measured on 2026-11-04 in the project virtual environment, Python 3.12.

- Baseline before changes: 74 Phase 1 tests passed. The Phase 2 prompt's claim
  of 143 existing tests did not match the files in this workspace.
- Final suite: 114 passed in 1.76 seconds; no Phase 1 files or APIs changed.
- Local OCR on six generated multiplier images: 6/6 recognized (`1.08x`,
  `2.45x`, `9.99x`, `10.00x`, `12.45x`, `25.67x`). This is synthetic-image
  recognition, not actual game accuracy.
- 30 captures of a 300 × 100 screen region: average capture 16.22 ms.
- Six synthetic OCR samples: average image processing plus OCR 137.83 ms,
  of which OCR was 135.24 ms. Combined average capture plus processing:
  154.04 ms. First-call model warm-up is included. These averages are
  environment-specific, not latency guarantees.
- One 1920 × 1080 monitor detected. Capture and monitor enumeration worked.
- Actual complete round detection latency is unmeasured: no calibrated game
  region or confirmed Firefox screenshot sequence was available.
- The live detector needs a visible phase transition and a stable blank or
  unreadable multiplier between results. If a site keeps the previous value
  visible through the next animation, the detector rejects that round instead
  of guessing. Real screenshots are needed to tune this behavior.

The runtime sends no website requests and saves screenshots only when the
diagnostic `--save` command is used. Its local RapidOCR models are bundled in
the installed package. Use `python -m examples.vision_benchmark` to rerun the
synthetic performance sample and `python -m examples.visual_diagnostic --monitors`
to inspect current displays.
