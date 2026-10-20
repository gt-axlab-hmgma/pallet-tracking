import numpy as np

from palletvision_track.kalman import KalmanFilter
from palletvision_track.track import Track, TrackIdCounter, TrackState

kf = KalmanFilter()


def new_track(box_xyah: tuple[float, float, float, float] = (100, 50, 2, 20)) -> Track:
    mean, cov = kf.initiate(np.array(box_xyah))
    return Track(mean[0], cov[0], score=0.9)


def state_of(track: Track) -> TrackState:
    """Read the state through a call, so mypy doesn't carry an earlier assert's narrowing over."""
    return track.state


def moved(track: Track) -> tuple[np.ndarray, np.ndarray]:
    """A different mean and covariance, so tests can see that update replaced them."""
    return track.mean + 1.0, track.cov * 2.0


def test_ids_start_at_one_and_never_repeat() -> None:
    ids = TrackIdCounter()
    assert [ids.next() for _ in range(4)] == [1, 2, 3, 4]


def test_each_counter_is_independent_and_reset_starts_over() -> None:
    a, b = TrackIdCounter(), TrackIdCounter()
    a.next()
    a.next()
    assert b.next() == 1
    a.reset()
    assert a.next() == 1


def test_a_fresh_track_is_new_and_has_no_id() -> None:
    t = new_track()
    assert t.state is TrackState.NEW
    assert t.track_id == 0
    assert not t.is_activated


def test_activate_mid_video_starts_tracked_but_unconfirmed() -> None:
    t = new_track()
    t.activate(track_id=5, frame_id=12, confirmed=False)
    assert t.state is TrackState.TRACKED
    assert (t.track_id, t.start_frame, t.frame_id, t.tracklet_len) == (5, 12, 12, 0)
    assert not t.is_activated


def test_activate_in_the_first_frame_is_confirmed_at_once() -> None:
    t = new_track()
    t.activate(track_id=1, frame_id=1, confirmed=True)
    assert t.is_activated


def test_second_match_confirms_the_track() -> None:
    t = new_track()
    t.activate(track_id=1, frame_id=12, confirmed=False)
    mean, cov = moved(t)
    t.update(mean, cov, score=0.7, frame_id=13)
    assert t.is_activated
    assert t.state is TrackState.TRACKED
    assert (t.frame_id, t.tracklet_len, t.score) == (13, 1, 0.7)
    assert t.start_frame == 12  # birth frame does not move
    np.testing.assert_array_equal(t.mean, mean)
    np.testing.assert_array_equal(t.cov, cov)


def test_tracklet_len_counts_consecutive_matches() -> None:
    t = new_track()
    t.activate(track_id=1, frame_id=1, confirmed=True)
    for frame in range(2, 6):
        t.update(t.mean, t.cov, score=0.9, frame_id=frame)
    assert t.tracklet_len == 4


def test_lost_then_re_activated() -> None:
    t = new_track()
    t.activate(track_id=1, frame_id=1, confirmed=True)
    t.update(t.mean, t.cov, score=0.9, frame_id=2)
    t.mark_lost()
    assert t.state is TrackState.LOST
    assert t.frame_id == 2  # still remembers when it was last seen
    mean, cov = moved(t)
    t.re_activate(mean, cov, score=0.6, frame_id=9)
    assert state_of(t) is TrackState.TRACKED
    assert t.is_activated
    assert (t.frame_id, t.tracklet_len, t.score, t.track_id) == (9, 0, 0.6, 1)
    np.testing.assert_array_equal(t.mean, mean)


def test_removed_is_a_state_and_keeps_the_id() -> None:
    t = new_track()
    t.activate(track_id=3, frame_id=4, confirmed=False)
    t.mark_removed()
    assert t.state is TrackState.REMOVED
    assert t.track_id == 3


def test_box_is_the_filtered_state_as_corners() -> None:
    # centre (100, 50), a = 2, h = 20 -> w = 40
    np.testing.assert_allclose(new_track().box, [80, 40, 120, 60])
