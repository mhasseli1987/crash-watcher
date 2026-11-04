"""Measure local capture and bundled OCR on synthetic multiplier text."""
from statistics import mean
import cv2
import numpy as np
from crash_watcher.vision.capture import CaptureConfig, ScreenCapture
from crash_watcher.vision.recognition import MultiplierRecognizer, RapidOCRReader

def main() -> None:
    with ScreenCapture(CaptureConfig(1, 0, 0, 300, 100)) as capture:
        captures = [capture.capture()[0] for _ in range(30)]
    recognizer = MultiplierRecognizer(RapidOCRReader())
    processing, ocr = [], []
    correct = 0
    for text in ("1.08x", "2.45x", "9.99x", "10.00x", "12.45x", "25.67x"):
        image = np.zeros((120, 360, 3), np.uint8)
        cv2.putText(image, text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        result = recognizer.recognize(image)
        correct += result.multiplier is not None and str(result.multiplier) + "x" == text
        processing.append(result.latency_ms)
        ocr.append(result.ocr_ms)
    print(f"capture_samples={len(captures)} synthetic_ocr={correct}/6 "
          f"capture_ms={mean(f.capture_ms for f in captures):.2f} "
          f"processing_ms={mean(processing):.2f} ocr_ms={mean(ocr):.2f} "
          f"total_ms={mean(f.capture_ms for f in captures)+mean(processing):.2f}")
    print("Synthetic text only; actual game OCR accuracy and round latency remain unmeasured.")

if __name__ == "__main__":
    main()
