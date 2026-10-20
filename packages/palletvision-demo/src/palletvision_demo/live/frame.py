import cv2
import numpy as np

from palletvision_infer.media.frame_source import FrameSource, parse_source

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def placeholder_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """A slanted floor grid, like an angled overhead camera, for demoing without footage."""
    img = np.full((height, width, 3), (46, 42, 38), np.uint8)
    horizon = int(height * 0.18)
    vanish = (width // 2, horizon - 40)
    for k in range(-12, 13):  # lines converging towards the vanishing point
        x = int(width / 2 + k * width / 10)
        cv2.line(img, (x, height), vanish, (88, 82, 74), 1, cv2.LINE_AA)
    for k in range(1, 14):  # floor rows, spaced closer towards the horizon
        y = int(horizon + (height - horizon) * (k / 13) ** 1.8)
        cv2.line(img, (0, y), (width, y), (88, 82, 74), 1, cv2.LINE_AA)
    cv2.putText(
        img, "no source yet: load a video clip or connect a camera", (24, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 144, 136), 1, cv2.LINE_AA,
    )  # fmt: skip
    return img


def load_reference_frame(source: str | None) -> np.ndarray:
    """Image file, first frame of a video, a webcam snapshot (digits), or the placeholder."""
    if source is None:
        return placeholder_frame()
    parsed = parse_source(source)
    if isinstance(parsed, str) and any(parsed.lower().endswith(s) for s in IMAGE_SUFFIXES):
        image = cv2.imread(parsed)
        if image is None:
            raise OSError(f"cannot read image: {parsed}")
        return np.asarray(image)
    with FrameSource(parsed) as frames:
        for i, frame in enumerate(frames):
            # a webcam needs a few frames to settle exposure; a video's first frame is fine
            if isinstance(parsed, str) or i >= 5:
                return frame.image
    raise OSError(f"no frames in source: {source}")


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError("could not encode frame")
    return bytes(buf)
