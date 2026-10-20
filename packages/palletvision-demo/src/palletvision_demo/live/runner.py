import re
import threading
import time

import numpy as np

from palletvision_demo.live.frame import encode_jpeg, load_reference_frame
from palletvision_demo.live.state import OBJECT_TYPE, DemoState
from palletvision_demo.settings import Settings
from palletvision_infer.detections import Detector
from palletvision_infer.media.frame_source import FrameSource
from palletvision_track.bytetrack import ByteTracker, TrackerConfig
from palletvision_zones.engine import Track

CAMERA_URL = re.compile(r"^(rtsp|rtsps|http|https)://\S+$", re.IGNORECASE)
CREDENTIALS = re.compile(r"(?<=://)[^/@\s]*@")


def redact(text: str) -> str:
    """Drop ``user:password@`` from any stream URL in ``text``, so it is never shown or logged."""
    return CREDENTIALS.sub("", text)


def camera_label(source: int | str) -> str:
    """What to call a camera in the UI: ``camera 0``, or a stream URL without its credentials."""
    return f"camera {source}" if isinstance(source, int) else redact(source)


def parse_camera(device: str) -> int | str:
    """A camera index (``0`` is the first one) or a network stream URL; nothing else.

    Paths are refused on purpose: a camera field that opened any file would let anyone who can
    reach the API read files off the machine.
    """
    value = device.strip()
    if value.isdigit():
        return int(value)
    if CAMERA_URL.match(value):
        return value
    raise ValueError("a camera is an index such as 0, or an rtsp:// or http(s):// stream URL")


def source_kind(value: str) -> str:
    """How ``--source`` is read at start-up: an index or a stream URL is a camera, else a file."""
    return "camera" if value.isdigit() or CAMERA_URL.match(value) else "video"


class LivePlayer:
    """Plays one source at a time, a video file or a camera, through detector and tracker.

    Each frame is detected, the detections go to ByteTrack, and the confirmed tracks seen in
    that frame are handed to the demo state, which runs the zone engine and logs the events. The
    frame itself is published for the browser to show behind the zones. Starting another source
    stops the current one first.
    """

    def __init__(self, state: DemoState, detector: Detector, settings: Settings) -> None:
        self._state = state
        self._detector = detector
        self._settings = settings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, source: int | str, kind: str, name: str) -> None:
        """Open ``source`` and start playing it. Raises OSError if it cannot be read."""
        self.stop()
        frame = load_reference_frame(str(source))  # fails here, before anything is replaced
        self._state.clear_tracks()
        self._state.set_reference(frame)
        self._state.publish_frame(encode_jpeg(frame, quality=80))
        self._state.set_source(kind, name, "playing")
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(source, kind, name, self._stop), name="live-player", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop playing. The last frame stays on screen and open visits stay open."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def eject(self) -> None:
        """Stop and forget the source: its tracks exit their zones."""
        self.stop()
        self._state.clear_tracks()
        self._state.clear_source()

    def _run(self, source: int | str, kind: str, name: str, stop: threading.Event) -> None:
        try:
            ran_out = self._play(source, kind, stop)
        except Exception as exc:  # shown in the UI; the API keeps serving what it has
            self._state.set_source(kind, name, "error", redact(str(exc)))
            return
        if not ran_out:
            return
        if kind == "camera":
            self._state.set_source(kind, name, "error", "The camera stopped sending frames.")
        else:
            self._state.set_source(kind, name, "ended")

    def _play(self, source: int | str, kind: str, stop: threading.Event) -> bool:
        """Returns True if the source ran out, False if it was stopped."""
        cfg = self._settings
        period = 1.0 / cfg.fps
        tracker = ByteTracker(
            TrackerConfig(max_time_lost=max(1, round(cfg.lost_seconds * cfg.fps)))
        )
        while True:
            frame_id = 0
            with FrameSource(source, fps=cfg.fps) as frames:
                for frame in frames:
                    if stop.is_set():
                        return False
                    began = time.perf_counter()
                    det = self._detector.detect(frame.image)
                    frame_id += 1
                    seen = tracker.update(frame_id, det.boxes, det.scores)
                    self._state.apply_tracks(
                        [Track(t.track_id, OBJECT_TYPE, _corners(t.box)) for t in seen],
                        tracker.removed_ids,
                    )
                    self._state.publish_frame(encode_jpeg(frame.image, quality=80))
                    # A file is read as fast as it decodes, so hold it to the target frame rate.
                    # A camera delivers in real time, and waiting would let its buffer go stale.
                    if kind == "video":
                        spare = period - (time.perf_counter() - began)
                        if stop.wait(max(spare, 0.0)):
                            return False
            if kind == "camera" or not cfg.loop:
                return not stop.is_set()
            if stop.is_set():
                return False
            tracker.reset()  # a new pass over the video: nothing carries over
            self._state.clear_tracks()


def _corners(box: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1, y1, x2, y2)
