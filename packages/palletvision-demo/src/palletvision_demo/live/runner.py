import threading
import time

import numpy as np

from palletvision_demo.live.frame import encode_jpeg
from palletvision_demo.live.state import DemoState
from palletvision_demo.settings import Settings
from palletvision_infer.detections import Detector
from palletvision_infer.media.frame_source import FrameSource, parse_source
from palletvision_track.bytetrack import ByteTracker, TrackerConfig
from palletvision_zones.engine import Track

# Stage 1 has one pallet class and the tracker ignores classes, so every track is the same type.
OBJECT_TYPE = "pallet_unknown"


class LiveRunner:
    """Plays a video through detector, tracker and zone engine on a background thread.

    Each frame is detected, the detections go to ByteTrack, and the confirmed tracks seen in
    that frame are handed to the demo state, which runs the zone engine and logs the events. The
    frame itself is published for the browser to show behind the zones.
    """

    def __init__(self, state: DemoState, detector: Detector, settings: Settings) -> None:
        assert settings.source is not None
        self._state = state
        self._detector = detector
        self._settings = settings
        self._tracker = ByteTracker(
            TrackerConfig(max_time_lost=max(1, round(settings.lost_seconds * settings.fps)))
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-runner", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        try:
            self._play()
        except Exception as exc:  # shown in the UI; the API keeps serving what it has
            self._state.add_warning(f"Live tracking stopped: {exc}")

    def _play(self) -> None:
        cfg = self._settings
        assert cfg.source is not None
        period = 1.0 / cfg.fps
        while not self._stop.is_set():
            frame_id = 0
            with FrameSource(parse_source(cfg.source), fps=cfg.fps) as frames:
                for frame in frames:
                    began = time.perf_counter()
                    det = self._detector.detect(frame.image)
                    frame_id += 1
                    seen = self._tracker.update(frame_id, det.boxes, det.scores)
                    self._state.apply_tracks(
                        [Track(t.track_id, OBJECT_TYPE, _corners(t.box)) for t in seen],
                        self._tracker.removed_ids,
                    )
                    self._state.publish_frame(encode_jpeg(frame.image, quality=80))
                    # a file is read as fast as it decodes, so hold it to the target frame rate
                    spare = period - (time.perf_counter() - began)
                    if self._stop.wait(max(spare, 0.0)):
                        return
            if not cfg.loop:
                return
            self._tracker.reset()  # a new pass over the video: nothing carries over
            self._state.clear_tracks()


def _corners(box: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1, y1, x2, y2)
