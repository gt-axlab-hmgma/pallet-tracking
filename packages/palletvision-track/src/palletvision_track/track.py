"""One tracked object: its Kalman state plus the bookkeeping that decides what it is allowed to do.

A track moves through four states:

    NEW ──activate──► TRACKED ──mark_lost──► LOST ──re_activate──► TRACKED
                         │                     │
                         └──────────┬──────────┘
                               mark_removed ──► REMOVED  (final)

The Kalman maths is not here. The tracker predicts and updates all tracks in one batch and hands
each track its new mean and covariance, so a track never calls the filter itself.
"""

from dataclasses import dataclass
from enum import Enum

from palletvision_track.boxes import Array, xyah_to_xyxy


class TrackState(Enum):
    NEW = 0  # created, not yet started
    TRACKED = 1  # seen in the latest frame
    LOST = 2  # not seen recently; kept alive in case the object reappears
    REMOVED = 3  # given up on; never comes back


class TrackIdCounter:
    """Hands out track IDs 1, 2, 3, ... and never reuses one.

    One counter per tracker, so two trackers don't share IDs and a reset starts again from 1.
    """

    def __init__(self) -> None:
        self._last = 0

    def next(self) -> int:
        self._last += 1
        return self._last

    def reset(self) -> None:
        self._last = 0


@dataclass
class Track:
    mean: Array  # (8,) Kalman state (x, y, a, h, vx, vy, va, vh)
    cov: Array  # (8, 8) Kalman covariance
    score: float  # detector score of the latest matched detection
    track_id: int = 0  # 0 until activated
    state: TrackState = TrackState.NEW
    # Only activated tracks are reported. A track born mid-video starts unactivated and has to be
    # matched once more, so a one-frame false positive never shows up in the output.
    is_activated: bool = False
    start_frame: int = 0
    frame_id: int = 0  # the last frame this track was matched (or born in)
    tracklet_len: int = 0  # matches since birth or since coming back from LOST

    @property
    def box(self) -> Array:
        """The filtered box as ``(x1, y1, x2, y2)``."""
        box: Array = xyah_to_xyxy(self.mean[None, :4])[0]
        return box

    def activate(self, track_id: int, frame_id: int, confirmed: bool) -> None:
        """Start the track. ``confirmed`` skips the wait for a second match.

        The tracker passes ``confirmed=True`` only in the first frame of the video, where there
        is no earlier frame to be confirmed against.
        """
        self.track_id = track_id
        self.state = TrackState.TRACKED
        self.is_activated = confirmed
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.tracklet_len = 0

    def update(self, mean: Array, cov: Array, score: float, frame_id: int) -> None:
        """A detection matched this track while it was being tracked."""
        self.mean, self.cov, self.score = mean, cov, score
        self.state = TrackState.TRACKED
        self.is_activated = True  # the second match confirms a new track
        self.frame_id = frame_id
        self.tracklet_len += 1

    def re_activate(self, mean: Array, cov: Array, score: float, frame_id: int) -> None:
        """A detection matched this track after it had been LOST."""
        self.mean, self.cov, self.score = mean, cov, score
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len = 0

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED
