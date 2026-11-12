"""Test OCR on current screen."""
import mss
import cv2
import numpy as np
from crash_watcher.desktop.config import load_settings
from crash_watcher.desktop.worker import ScreenUnavailable

def test_capture():
    """Test screen capture with current settings."""
    settings = load_settings()

    if settings.result_region is None:
        print("ERROR: result_region not set")
        print("Run calibration first to select the region.")
        return

    print(f"Result region: {settings.result_region}")

    try:
        import mss
        from crash_watcher.vision.capture import ScreenCapture

        with ScreenCapture(settings.result_region.capture(100)) as capture:
            from mss import mss as mss_backend
            with mss_backend() as sct:
                monitor = sct.monitors[settings.result_region.monitor]
                print(f"Monitor {settings.result_region.monitor}: {monitor}")

                # Try direct capture
                for mon_idx in range(1, len(sct.monitors)):
                    m = sct.monitors[mon_idx]
                    print(f"Available monitor {mon_idx}: left={m['left']}, top={m['top']}, width={m['width']}, height={m['height']}")

                # Capture the region
                raw = sct.grab({
                    "left": settings.result_region.x,
                    "top": settings.result_region.y,
                    "width": settings.result_region.width,
                    "height": settings.result_region.height
                })
                frame = np.asarray(raw)[:, :, :3].copy()

                print(f"Captured frame shape: {frame.shape}")
                print(f"Pixel at center: {frame[frame.shape[0]//2, frame.shape[1]//2]}")

                # Show the captured region
                cv2.imwrite("test_capture.png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print("Saved test_capture.png")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_capture()