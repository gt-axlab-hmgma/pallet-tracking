import numpy as np

from palletvision_track.bytetrack import ByteTracker, TrackerConfig
from palletvision_track.kalman import KalmanFilter
from palletvision_track.track import Track, TrackState

W, H = 40.0, 80.0


def box(cx: float, cy: float = 100.0) -> list[float]:
    return [cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2]


def run(
    tracker: ByteTracker, frame: int, objects: list[tuple[float, float]]
) -> dict[int, list[float]]:
    """Feed one frame of ``(centre x, score)`` objects; return {track id: box}."""
    boxes = [box(cx) for cx, _ in objects]
    scores = [s for _, s in objects]
    return {t.track_id: t.box.tolist() for t in tracker.update(frame, boxes, scores)}


def ids(tracks: list[Track]) -> list[int]:
    return [t.track_id for t in tracks]


def test_first_frame_tracks_are_reported_at_once() -> None:
    tracker = ByteTracker()
    assert sorted(run(tracker, 1, [(100, 0.9), (400, 0.9)])) == [1, 2]


def test_a_moving_object_keeps_its_id() -> None:
    tracker = ByteTracker()
    seen = {tuple(run(tracker, f, [(100 + 5 * f, 0.9)])) for f in range(1, 30)}
    assert seen == {(1,)}


def test_a_track_born_mid_video_is_reported_only_after_a_second_match() -> None:
    tracker = ByteTracker()
    run(tracker, 1, [(100, 0.9)])
    assert sorted(run(tracker, 2, [(100, 0.9), (400, 0.9)])) == [1]  # new box, not yet reported
    assert sorted(run(tracker, 3, [(100, 0.9), (400, 0.9)])) == [1, 2]  # confirmed


def test_a_one_frame_false_positive_is_never_reported() -> None:
    tracker = ByteTracker()
    run(tracker, 1, [(100, 0.9)])
    run(tracker, 2, [(100, 0.9), (700, 0.9)])  # flash of a box
    for f in range(3, 10):
        assert sorted(run(tracker, f, [(100, 0.9)])) == [1]
    # and it is gone, not waiting: a real box at that spot gets the next id
    run(tracker, 10, [(100, 0.9), (700, 0.9)])
    assert sorted(run(tracker, 11, [(100, 0.9), (700, 0.9)])) == [1, 3]


def test_a_weak_high_score_box_cannot_start_a_track() -> None:
    # 0.65 is "high" (> 0.6) so it can match tracks, but starting one needs > 0.7
    tracker = ByteTracker()
    run(tracker, 1, [(100, 0.9)])
    for f in range(2, 6):
        assert sorted(run(tracker, f, [(100, 0.9), (400, 0.65)])) == [1]


def test_low_score_box_keeps_a_track_alive_through_the_second_round() -> None:
    tracker = ByteTracker()
    for f in range(1, 4):
        run(tracker, f, [(100, 0.9)])
    # the object is partly hidden: score drops to 0.3, still the same place
    assert sorted(run(tracker, 4, [(100, 0.3)])) == [1]
    assert sorted(run(tracker, 5, [(100, 0.9)])) == [1]


def test_low_score_box_that_matches_nothing_is_ignored() -> None:
    tracker = ByteTracker()
    run(tracker, 1, [(100, 0.9)])
    for f in range(2, 8):
        assert sorted(run(tracker, f, [(100, 0.9), (500, 0.3)])) == [1]


def test_a_lost_track_comes_back_with_its_id() -> None:
    tracker = ByteTracker()
    for f in range(1, 4):
        run(tracker, f, [(100, 0.9)])
    for f in range(4, 9):  # hidden for 5 frames
        assert run(tracker, f, []) == {}
    assert sorted(run(tracker, 9, [(100, 0.9)])) == [1]


def test_a_lost_track_is_not_recovered_by_a_low_score_box() -> None:
    # official behaviour: the second round is only for tracks that were seen last frame
    tracker = ByteTracker()
    for f in range(1, 4):
        run(tracker, f, [(100, 0.9)])
    run(tracker, 4, [])  # now lost
    assert run(tracker, 5, [(100, 0.3)]) == {}
    assert sorted(run(tracker, 6, [(100, 0.9)])) == [1]  # a confident box does


def test_a_track_lost_longer_than_max_time_lost_is_dropped() -> None:
    tracker = ByteTracker(TrackerConfig(max_time_lost=3))
    for f in range(1, 4):
        run(tracker, f, [(100, 0.9)])
    for f in range(4, 9):  # last seen frame 3; removed once frame - 3 > 3, i.e. at frame 7
        run(tracker, f, [])
    run(tracker, 9, [(100, 0.9)])
    assert sorted(run(tracker, 10, [(100, 0.9)])) == [2]  # a new id, not 1


def test_lost_track_still_alive_at_the_limit() -> None:
    tracker = ByteTracker(TrackerConfig(max_time_lost=3))
    for f in range(1, 4):
        run(tracker, f, [(100, 0.9)])
    for f in range(4, 7):  # frame 6 is exactly 3 frames after the last sighting
        run(tracker, f, [])
    assert sorted(run(tracker, 7, [(100, 0.9)])) == [1]


def test_ids_do_not_depend_on_detection_order() -> None:
    a, b = ByteTracker(), ByteTracker()
    run(a, 1, [(100, 0.9), (400, 0.9)])
    run(b, 1, [(100, 0.9), (400, 0.9)])
    ra = run(a, 2, [(104, 0.9), (404, 0.9)])
    rb = run(b, 2, [(404, 0.9), (104, 0.9)])
    assert ra == rb


def test_two_objects_keep_separate_ids() -> None:
    tracker = ByteTracker()
    for f in range(1, 20):
        out = run(tracker, f, [(100 + 3 * f, 0.9), (600 - 3 * f, 0.9)])
        assert sorted(out) == [1, 2]
        assert out[1][0] < out[2][0]  # id 1 stays the left one


def test_empty_frames_do_not_crash_and_empty_start_is_fine() -> None:
    tracker = ByteTracker()
    assert run(tracker, 1, []) == {}
    assert run(tracker, 2, []) == {}
    assert sorted(run(tracker, 3, [(100, 0.9)])) == []  # first sighting, mid-video: unconfirmed
    assert sorted(run(tracker, 4, [(100, 0.9)])) == [1]


def test_reset_restarts_ids_and_first_frame_rule() -> None:
    tracker = ByteTracker()
    run(tracker, 1, [(100, 0.9)])
    run(tracker, 2, [(100, 0.9)])
    tracker.reset()
    assert sorted(run(tracker, 1, [(300, 0.9)])) == [1]


def test_output_boxes_follow_the_filtered_motion() -> None:
    tracker = ByteTracker()
    out: dict[int, list[float]] = {}
    for f in range(1, 40):
        out = run(tracker, f, [(100 + 5 * f, 0.9)])
    np.testing.assert_allclose(out[1], box(100 + 5 * 39), atol=1.0)


def test_a_score_exactly_at_high_thresh_is_neither_high_nor_low() -> None:
    # official: high is > thresh and low is < thresh, so 0.6 is ignored altogether
    tracker = ByteTracker()
    run(tracker, 1, [(100, 0.9)])
    assert run(tracker, 2, [(100, 0.6)]) == {}  # no high box to match, no low box either


def test_a_score_exactly_at_new_track_thresh_starts_a_track() -> None:
    tracker = ByteTracker()
    run(tracker, 1, [(100, 0.9)])
    run(tracker, 2, [(100, 0.9), (400, 0.7)])
    assert sorted(run(tracker, 3, [(100, 0.9), (400, 0.9)])) == [1, 2]


def _planted(state: TrackState, start: int, last: int, tid: int) -> Track:
    mean, cov = KalmanFilter().initiate(np.array([100.0, 100.0, 0.5, 80.0]))
    t = Track(mean[0], cov[0], score=0.9, track_id=tid, state=state, is_activated=True)
    t.start_frame, t.frame_id = start, last
    return t


def test_a_lost_track_on_top_of_a_tracked_one_is_dropped_if_younger() -> None:
    tracker = ByteTracker()
    old = _planted(TrackState.TRACKED, start=1, last=20, tid=1)
    young = _planted(TrackState.LOST, start=10, last=18, tid=2)
    tracker._tracks = [old, young]
    tracker._drop_duplicates()
    assert tracker._tracks == [old]
    assert young.state is TrackState.REMOVED


def test_a_tracked_track_on_top_of_an_older_lost_one_is_dropped() -> None:
    tracker = ByteTracker()
    young = _planted(TrackState.TRACKED, start=10, last=20, tid=1)
    old = _planted(TrackState.LOST, start=1, last=18, tid=2)
    tracker._tracks = [young, old]
    tracker._drop_duplicates()
    assert tracker._tracks == [old]


def test_tracked_and_lost_tracks_in_different_places_are_both_kept() -> None:
    tracker = ByteTracker()
    a = _planted(TrackState.TRACKED, start=1, last=20, tid=1)
    b = _planted(TrackState.LOST, start=1, last=18, tid=2)
    b.mean = b.mean + np.array([200.0, 0, 0, 0, 0, 0, 0, 0])
    tracker._tracks = [a, b]
    tracker._drop_duplicates()
    assert len(tracker._tracks) == 2


def test_removed_ids_lists_tracks_given_up_on_in_that_update() -> None:
    tracker = ByteTracker(TrackerConfig(max_time_lost=3))
    for f in range(1, 4):
        run(tracker, f, [(100, 0.9)])
    for f in range(4, 8):  # last seen in frame 3; removed in frame 7 (7 - 3 > 3)
        run(tracker, f, [])
        assert tracker.removed_ids == ([1] if f == 7 else [])
    run(tracker, 8, [])
    assert tracker.removed_ids == []  # reported once


def test_a_track_that_never_confirmed_is_reported_removed_too() -> None:
    # harmless for callers (they never saw it), but the list is complete
    tracker = ByteTracker()
    run(tracker, 1, [(100, 0.9)])
    run(tracker, 2, [(100, 0.9), (700, 0.9)])
    run(tracker, 3, [(100, 0.9)])  # the flash is not seen again
    assert tracker.removed_ids == [2]
