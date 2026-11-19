"""Local CLI calibration, screenshots, and live monitoring."""
import argparse
from pathlib import Path
from threading import Lock
from time import sleep
import cv2
import mss
from crash_watcher import StrategyConfig, StrategyEngine
from crash_watcher.vision.capture import CaptureConfig, ScreenCapture
from crash_watcher.vision.detection import DetectorConfig, RoundDetector
from crash_watcher.vision.pipeline import VisualPipeline
from crash_watcher.vision.recognition import MultiplierRecognizer, RapidOCRReader, tesseract_reader

def main() -> None:
    parser = argparse.ArgumentParser(description="Local Crash Watcher vision diagnostics")
    parser.add_argument("--monitors", action="store_true")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--tesseract", help="Path to local tesseract.exe")
    parser.add_argument("--save", action="store_true", help="Save one screenshot in local dataset")
    parser.add_argument("--preview", action="store_true", help="Show captured region with OpenCV")
    parser.add_argument("--dataset", default="dataset")
    args = parser.parse_args()
    if args.monitors:
        with mss.MSS() as backend:
            for index, monitor in enumerate(backend.monitors[1:], 1):
                print(index, dict(monitor))
        return
    if None in (args.x, args.y, args.width, args.height):
        parser.error("--x, --y, --width and --height are required")
    if args.tesseract:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = args.tesseract
        try:
            pytesseract.get_tesseract_version()
        except (pytesseract.TesseractNotFoundError, OSError):
            parser.error("Tesseract OCR is unavailable at the given path")
        reader = tesseract_reader
    else:
        reader = RapidOCRReader()
    config = CaptureConfig(args.monitor, args.x, args.y, args.width, args.height, args.interval_ms)
    capture = ScreenCapture(config)
    if args.save:
        with capture:
            frame, _ = capture.capture()
        directory = Path(args.dataset).resolve()
        if not directory.is_relative_to(Path.cwd().resolve()):
            parser.error("dataset must be inside the project directory")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"sample-{frame.captured_at.strftime('%Y%m%d-%H%M%S-%f')}.png"
        cv2.imwrite(str(path), frame.image)
        print(path)
        return
    lock = Lock()
    latest = [None]
    def output(item):
        with lock:
            latest[0] = item
        r = item.detection.recognition
        if r:
            print(f"state={item.detection.state.value} value={r.multiplier} category={r.category} "
                  f"color={r.color_category} confidence={r.confidence:.2f} ocr_ms={r.ocr_ms:.1f}")
        if item.strategy:
            print(f"CONFIRMED {item.strategy.round_id} {item.strategy.multiplier} strategy={item.strategy.current_state.value} "
                  f"events={[e.event_type.value for e in item.strategy.events]}")
    pipeline = VisualPipeline(capture, MultiplierRecognizer(reader), RoundDetector(DetectorConfig()),
                              StrategyEngine(StrategyConfig()), output, lambda e: print(e.kind.value, e.detail))
    window = None
    closed = [False]
    if args.preview:
        import tkinter as tk
        from PIL import Image, ImageTk
        window = tk.Tk()
        window.title("Crash Watcher diagnostic region")
        label = tk.Label(window)
        label.pack()
        window.protocol("WM_DELETE_WINDOW", lambda: closed.__setitem__(0, True))
        window.bind("q", lambda _: closed.__setitem__(0, True))
    pipeline.start()
    try:
        while True:
            if args.preview:
                with lock:
                    item = latest[0]
                # Preview is intentionally a separate capture for development only.
                with mss.MSS() as backend:
                    raw = backend.grab({"left": args.x, "top": args.y, "width": args.width, "height": args.height})
                    import numpy as np
                    rgb = cv2.cvtColor(np.asarray(raw)[:, :, :3], cv2.COLOR_BGR2RGB)
                    photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                    label.configure(image=photo)
                    label.image = photo
                window.update()
                if closed[0]:
                    break
                sleep(0.1)
            else:
                sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if window is not None:
            window.destroy()
        print("Average latency (ms):", pipeline.metrics.report())

if __name__ == "__main__":
    main()
