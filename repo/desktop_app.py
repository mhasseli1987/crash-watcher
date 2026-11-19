"""Standalone GUI entry point for Python and PyInstaller."""
import json
import sys
from pathlib import Path

def self_test() -> int:
    """Exercise bundled OCR without opening a window; write a local report."""
    import cv2
    import numpy as np
    from crash_watcher.desktop.final_result import BangRecognizer
    from crash_watcher.vision.recognition import RapidOCRReader
    image = np.zeros((120, 500, 3), np.uint8)
    cv2.putText(image, "Bang @13.05x", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    reading = BangRecognizer(RapidOCRReader()).recognize(image)
    passed = str(reading.value) == "13.05"
    target = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    (target / "self-test.json").write_text(json.dumps({"bundled_ocr": passed, "text": reading.raw_text}), encoding="utf-8")
    return 0 if passed else 1

if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(self_test())
    from crash_watcher.desktop.__main__ import main
    raise SystemExit(main())
