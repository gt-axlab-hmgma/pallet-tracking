"""ByteTrack (Zhang et al., ECCV 2022, Algorithm 1 "BYTE") on top of a Kalman filter and IoU.

The idea: keep every detection, not only the confident ones. Confident boxes are matched to
tracks first. Tracks left over (usually an object that is partly hidden, so its score dropped) then
get a second chance against the low-score boxes. Low-score boxes that match nothing are dropped
as background.

The paper's pseudo-code leaves out details that the official code (github.com/ifzhang/ByteTrack,
``yolox/tracker/byte_tracker.py``, MIT) has, and this follows the official code: tracks born last
frame wait for a second match, a new track needs a higher score than a matching one, lost tracks
are kept for ``max_time_lost`` frames, lost tracks are not offered low-score boxes, and a lost
track that duplicates a tracked one is dropped.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from palletvision_track.boxes import Array, xyah_to_xyxy, xyxy_to_xyah
from palletvision_track.kalman import KalmanFilter
from palletvision_track.matching import IntArray, assign, iou_distance
from palletvision_track.track import Track, TrackIdCounter, TrackState


@dataclass(frozen=True)
class TrackerConfig:
    """Thresholds. Scores are detector confidences, costs are ``1 - IoU`` (lower is closer)."""

    high_thresh: float = 0.6  # a detection above this is "high score"
    low_thresh: float = 0.1  # between this and high_thresh is "low score"; below is dropped
    match_thresh: float = 0.8  # first round: max cost to match a track to a high-score box
    second_thresh: float = 0.5  # second round: max cost to match a track to a low-score box
    unconfirmed_thresh: float = 0.7  # max cost to confirm a track born last frame
    max_time_lost: int = 30  # frames a lost track is kept before it is given up on
    fuse_score: bool = True  # multiply IoU by the detection score in the first and third match

    @property
    def new_track_thresh(self) -> float:
        """A detection must score above this to start a track, stricter than to match one."""
        return self.high_thresh + 0.1


# Two boxes with IoU above 0.85 are the same object.
_DUPLICATE_COST = 0.15


class ByteTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._kf = KalmanFilter()
        self._ids = TrackIdCounter()
        self._tracks: list[Track] = []  # TRACKED and LOST ones, including unconfirmed
        self._started = False
        # IDs of tracks the last update gave up on, for callers that keep state per track
        self.removed_ids: list[int] = []

    def reset(self) -> None:
        self._ids.reset()
        self._tracks = []
        self._started = False
        self.removed_ids = []

    def update(self, frame_id: int, boxes: npt.ArrayLike, scores: npt.ArrayLike) -> list[Track]:
        """Feed one frame's detections ``(N, 4)`` corners and ``(N,)`` scores.

        ``frame_id`` must grow by 1 for every frame fed in, because ``max_time_lost`` counts
        frames. Returns the confirmed tracks seen in this frame. These are live objects that the
        next call changes, so copy what you need.
        """
        cfg = self.config
        boxes_ = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores_ = np.asarray(scores, dtype=np.float64).reshape(-1)
        first_frame = not self._started
        self._started = True

        # 1. Split the detections by score (Algorithm 1, lines 4-13).
        high = scores_ > cfg.high_thresh
        low = (scores_ > cfg.low_thresh) & (scores_ < cfg.high_thresh)
        d_high, s_high = boxes_[high], scores_[high]
        d_low, s_low = boxes_[low], scores_[low]

        # Tracks seen at least twice, alive or lost, form the pool. Tracks born last frame have
        # not been confirmed yet and are handled on their own.
        pool = [t for t in self._tracks if t.is_activated]
        unconfirmed = [t for t in self._tracks if not t.is_activated]

        # 2. Move the pool one frame forward (lines 14-16). Tracks born last frame are matched
        #    where they were first seen, as in the official code; they have no velocity yet.
        self._predict(pool)

        # 3. First association: the whole pool, lost tracks included, against high-score boxes
        #    (lines 17-19).
        cost = iou_distance(_boxes(pool), d_high, s_high if cfg.fuse_score else None)
        matches, u_pool, u_high = assign(cost, cfg.match_thresh)
        self._apply(pool, matches, d_high, s_high, frame_id)

        # 4. Second association (the key of the paper, lines 20-21): tracks that were being
        #    tracked and found nothing above against the low-score boxes, by IoU alone. A lost
        #    track is not offered low-score boxes, as in the official code.
        left = [pool[i] for i in u_pool if pool[i].state is TrackState.TRACKED]
        matches, u_left, _ = assign(iou_distance(_boxes(left), d_low), cfg.second_thresh)
        self._apply(left, matches, d_low, s_low, frame_id)
        for i in u_left:
            left[i].mark_lost()

        # 5. Tracks born last frame take what is left of the high-score boxes. One that finds
        #    nothing is a false positive and is removed.
        rest = d_high[u_high]
        cost = iou_distance(_boxes(unconfirmed), rest, s_high[u_high] if cfg.fuse_score else None)
        matches, u_unconfirmed, u_rest = assign(cost, cfg.unconfirmed_thresh)
        self._apply(unconfirmed, matches, rest, s_high[u_high], frame_id)
        for i in u_unconfirmed:
            unconfirmed[i].mark_removed()

        # 6. Strong boxes nobody claimed start new tracks (lines 23-25).
        born = self._spawn(rest[u_rest], s_high[u_high][u_rest], frame_id, first_frame)

        # 7. Give up on tracks that have been lost for too long.
        for t in self._tracks:
            if t.state is TrackState.LOST and frame_id - t.frame_id > cfg.max_time_lost:
                t.mark_removed()

        before = {t.track_id for t in self._tracks}
        self._tracks = [t for t in self._tracks + born if t.state is not TrackState.REMOVED]
        self._drop_duplicates()
        self.removed_ids = sorted(before - {t.track_id for t in self._tracks})
        # 8. Report confirmed tracks that were matched this frame. Lost tracks are not output.
        return [t for t in self._tracks if t.state is TrackState.TRACKED and t.is_activated]

    def _drop_duplicates(self) -> None:
        """If a tracked and a lost track sit on the same box, keep the one that lived longer."""
        tracked = [t for t in self._tracks if t.state is TrackState.TRACKED]
        lost = [t for t in self._tracks if t.state is TrackState.LOST]
        if not tracked or not lost:
            return
        dist = iou_distance(_boxes(tracked), _boxes(lost))
        for p, q in zip(*np.nonzero(dist < _DUPLICATE_COST), strict=True):
            # a tie drops the tracked one, as in the official code
            if (
                tracked[p].frame_id - tracked[p].start_frame
                > lost[q].frame_id - lost[q].start_frame
            ):
                lost[q].mark_removed()
            else:
                tracked[p].mark_removed()
        self._tracks = [t for t in self._tracks if t.state is not TrackState.REMOVED]

    def _predict(self, tracks: list[Track]) -> None:
        if not tracks:
            return
        mean = np.stack([t.mean for t in tracks])
        cov = np.stack([t.cov for t in tracks])
        # A lost track has no detection to correct its height velocity, so stop it growing or
        # shrinking while it is unseen (the official code does the same).
        not_tracked = np.array([t.state is not TrackState.TRACKED for t in tracks])
        mean[not_tracked, 7] = 0.0
        mean, cov = self._kf.predict(mean, cov)
        for t, m, c in zip(tracks, mean, cov, strict=True):
            t.mean, t.cov = m, c

    def _apply(
        self,
        tracks: list[Track],
        matches: IntArray,
        det_boxes: Array,
        det_scores: Array,
        frame_id: int,
    ) -> None:
        """Correct each matched track with its detection, in one batched Kalman update."""
        if len(matches) == 0:
            return
        matched = [tracks[i] for i in matches[:, 0]]
        det = matches[:, 1]
        mean, cov = self._kf.update(
            np.stack([t.mean for t in matched]),
            np.stack([t.cov for t in matched]),
            xyxy_to_xyah(det_boxes[det]),
        )
        for t, m, c, score in zip(matched, mean, cov, det_scores[det], strict=True):
            if t.state is TrackState.LOST:
                t.re_activate(m, c, float(score), frame_id)
            else:
                t.update(m, c, float(score), frame_id)

    def _spawn(self, boxes: Array, scores: Array, frame_id: int, first_frame: bool) -> list[Track]:
        strong = scores >= self.config.new_track_thresh
        if not strong.any():
            return []
        mean, cov = self._kf.initiate(xyxy_to_xyah(boxes[strong]))
        born = [Track(m, c, float(s)) for m, c, s in zip(mean, cov, scores[strong], strict=True)]
        for t in born:
            # In the first frame there is nothing earlier to wait for, so tracks count at once.
            t.activate(self._ids.next(), frame_id, confirmed=first_frame)
        return born


def _boxes(tracks: list[Track]) -> Array:
    """The tracks' current boxes as ``(N, 4)`` corners."""
    if not tracks:
        return np.zeros((0, 4))
    return xyah_to_xyxy(np.stack([t.mean[:4] for t in tracks]))
