"""
People detection on RTSP stream.
Apple Vision framework (no model download) — runs on Neural Engine / GPU on Apple Silicon.
"""
import av
import cv2
import numpy as np
import Quartz
import Vision

# ── tuneable parameters ───────────────────────────────────────────────────────

RTSP_URL = "rtsp://admin:admin123@192.168.1.169:554/cam/realmonitor?channel=1&subtype=0"

DETECT_EVERY_N_FRAMES = 10    # run Vision on every Nth frame (1 = every frame, higher = faster display)
CONF_THRESH           = 0.5  # minimum confidence to show a detection (0.0–1.0)
DETECT_WIDTH          = 640  # resize frame to this width before Vision inference (smaller = faster)

# ── module-level singletons (created once, reused every frame) ────────────────

_COLOR_SPACE = Quartz.CGColorSpaceCreateDeviceRGB()
_VN_REQUEST  = Vision.VNDetectHumanRectanglesRequest.alloc().init()


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_cgimage(img_bgr: np.ndarray):
    """Convert BGR numpy array to CGImage without copying via tobytes() when possible."""
    h, w  = img_bgr.shape[:2]
    # Vision needs RGB; make contiguous so the buffer pointer is stable
    rgb   = np.ascontiguousarray(img_bgr[:, :, ::-1])
    data  = rgb.tobytes()
    prov  = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    return Quartz.CGImageCreate(
        w, h, 8, 24, w * 3, _COLOR_SPACE,
        Quartz.kCGBitmapByteOrderDefault | Quartz.kCGImageAlphaNone,
        prov, None, False, Quartz.kCGRenderingIntentDefault,
    )


def detect_people(img_bgr: np.ndarray, orig_w: int, orig_h: int):
    """
    Detect people in img_bgr (possibly downscaled) and return boxes
    scaled back to (orig_w, orig_h).
    Returns list of (x1, y1, x2, y2, conf).
    """
    h, w  = img_bgr.shape[:2]
    sx    = orig_w / w
    sy    = orig_h / h

    cgimg   = _to_cgimage(img_bgr)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimg, {})
    ok, _   = handler.performRequests_error_([_VN_REQUEST], None)
    if not ok:
        return []

    out = []
    for obs in (_VN_REQUEST.results() or []):
        conf = float(obs.confidence())
        if conf < CONF_THRESH:
            continue
        bb = obs.boundingBox()                   # normalised, origin = bottom-left
        x1 = int(bb.origin.x * orig_w)
        y1 = int((1.0 - bb.origin.y - bb.size.height) * orig_h)
        x2 = int((bb.origin.x + bb.size.width)  * orig_w)
        y2 = int((1.0 - bb.origin.y)             * orig_h)
        out.append((x1, y1, x2, y2, conf))
    return out


def draw_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, conf: float) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 0), 2)
    label = f"Person {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 170, 0), -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    print(f"Detect every {DETECT_EVERY_N_FRAMES} frame(s) | "
          f"conf ≥ {CONF_THRESH} | detect width {DETECT_WIDTH}px")

    options = {"rtsp_transport": "tcp", "stimeout": "5000000", "max_delay": "500000"}
    try:
        container = av.open(RTSP_URL, options=options, timeout=10)
    except Exception as e:
        print(f"Cannot connect: {e}")
        return

    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    print("Connected. Press 'q' or ESC to quit.")

    frame_idx  = 0
    detections = []

    try:
        for av_frame in container.decode(stream):
            img    = av_frame.to_ndarray(format="bgr24")
            orig_h, orig_w = img.shape[:2]
            frame_idx += 1

            if frame_idx % DETECT_EVERY_N_FRAMES == 0:
                # Downscale for faster inference, keep aspect ratio
                scale   = DETECT_WIDTH / orig_w
                det_img = cv2.resize(img, (DETECT_WIDTH, int(orig_h * scale)),
                                     interpolation=cv2.INTER_LINEAR)
                detections = detect_people(det_img, orig_w, orig_h)

            for (x1, y1, x2, y2, conf) in detections:
                draw_box(img, x1, y1, x2, y2, conf)

            cv2.putText(img, f"People: {len(detections)}", (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 230), 2, cv2.LINE_AA)

            cv2.imshow("Camera — People Detection", img)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    except Exception as e:
        print(f"Stream error: {e}")
    finally:
        container.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
