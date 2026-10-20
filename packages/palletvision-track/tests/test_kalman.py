import numpy as np
import pytest

from palletvision_track.kalman import KalmanFilter

kf = KalmanFilter()


def test_initiate_puts_measurement_in_position_and_zero_in_velocity() -> None:
    mean, cov = kf.initiate(np.array([100.0, 50.0, 0.5, 80.0]))
    np.testing.assert_array_equal(mean, [[100, 50, 0.5, 80, 0, 0, 0, 0]])
    assert cov.shape == (1, 8, 8)


def test_initiate_covariance_matches_the_formula() -> None:
    h = 80.0
    _, cov = kf.initiate(np.array([0.0, 0.0, 1.0, h]))
    w_p, w_v = 1 / 20, 1 / 160
    expected_std = [2 * w_p * h, 2 * w_p * h, 1e-2, 2 * w_p * h]
    expected_std += [10 * w_v * h, 10 * w_v * h, 1e-5, 10 * w_v * h]
    np.testing.assert_allclose(np.diag(cov[0]), np.square(expected_std))
    np.testing.assert_array_equal(cov[0], np.diag(np.diag(cov[0])))


def test_noise_scales_with_height() -> None:
    _, small = kf.initiate(np.array([0.0, 0.0, 1.0, 10.0]))
    _, big = kf.initiate(np.array([0.0, 0.0, 1.0, 100.0]))
    # position variance is (std)^2, so 10x the height is 100x the variance
    assert big[0, 0, 0] / small[0, 0, 0] == pytest.approx(100)
    # the aspect-ratio noise has no height scale
    assert big[0, 2, 2] == small[0, 2, 2]


def test_predict_k_steps_moves_box_by_k_times_velocity() -> None:
    mean, cov = kf.initiate(np.array([100.0, 50.0, 0.5, 80.0]))
    mean[0, 4:] = [3.0, -2.0, 0.0, 1.0]  # vx, vy, va, vh
    for _ in range(5):
        mean, cov = kf.predict(mean, cov)
    np.testing.assert_allclose(mean[0, :4], [100 + 15, 50 - 10, 0.5, 80 + 5])
    np.testing.assert_allclose(mean[0, 4:], [3, -2, 0, 1])  # velocity itself is unchanged


def test_predict_grows_uncertainty() -> None:
    mean, cov = kf.initiate(np.array([0.0, 0.0, 1.0, 50.0]))
    _, cov2 = kf.predict(mean, cov)
    assert np.all(np.diag(cov2[0]) > np.diag(cov[0]))


def test_constant_velocity_track_converges() -> None:
    v = np.array([4.0, -1.0, 0.0, 0.0])
    z0 = np.array([100.0, 200.0, 0.5, 60.0])
    mean, cov = kf.initiate(z0)
    for t in range(1, 60):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, z0 + v * t)
    np.testing.assert_allclose(mean[0, 4:], v, atol=1e-2)  # velocity learned from positions alone
    mean, cov = kf.predict(mean, cov)
    np.testing.assert_allclose(mean[0, :4], z0 + v * 60, atol=1e-2)  # next position predicted


def test_update_pulls_mean_towards_the_measurement() -> None:
    mean, cov = kf.initiate(np.array([100.0, 100.0, 1.0, 50.0]))
    mean, cov = kf.predict(mean, cov)
    new_mean, new_cov = kf.update(mean, cov, np.array([110.0, 100.0, 1.0, 50.0]))
    assert 100 < new_mean[0, 0] < 110
    assert new_mean[0, 4] > 0  # an unexpected move to the right starts a rightward velocity
    assert new_cov[0, 0, 0] < cov[0, 0, 0]  # a measurement always shrinks uncertainty


def test_update_with_the_predicted_box_leaves_mean_unchanged() -> None:
    mean, cov = kf.initiate(np.array([100.0, 100.0, 1.0, 50.0]))
    mean, cov = kf.predict(mean, cov)
    new_mean, _ = kf.update(mean, cov, mean[:, :4])
    np.testing.assert_allclose(new_mean, mean)


def test_covariance_stays_symmetric_positive_definite() -> None:
    rng = np.random.default_rng(0)
    mean, cov = kf.initiate(np.array([100.0, 200.0, 0.5, 60.0]))
    for _ in range(200):
        mean, cov = kf.predict(mean, cov)
        if rng.random() < 0.7:  # miss a few frames like a real detector
            mean, cov = kf.update(mean, cov, mean[:, :4] + rng.normal(0, 1, (1, 4)))
        np.testing.assert_allclose(cov, cov.transpose(0, 2, 1), atol=1e-12)
        assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_batch_matches_one_track_at_a_time() -> None:
    zs = np.array([[100.0, 50.0, 0.5, 80.0], [300.0, 120.0, 1.5, 20.0], [10.0, 10.0, 1.0, 200.0]])
    new = zs + np.array([[2.0, -1.0, 0.0, 1.0]] * 3)
    mean, cov = kf.initiate(zs)
    mean, cov = kf.predict(mean, cov)
    batch_mean, batch_cov = kf.update(mean, cov, new)
    for i in range(3):
        m, c = kf.initiate(zs[i])
        m, c = kf.predict(m, c)
        m, c = kf.update(m, c, new[i])
        np.testing.assert_allclose(batch_mean[i], m[0])
        np.testing.assert_allclose(batch_cov[i], c[0])


def test_empty_batch() -> None:
    mean, cov = kf.initiate(np.zeros((0, 4)))
    assert mean.shape == (0, 8)
    assert kf.predict(mean, cov)[0].shape == (0, 8)
    assert kf.update(mean, cov, np.zeros((0, 4)))[0].shape == (0, 8)


def test_negative_predicted_height_does_not_break_the_noise() -> None:
    mean, cov = kf.initiate(np.array([0.0, 0.0, 1.0, 1.0]))
    mean[0, 7] = -5.0  # height shrinking fast
    for _ in range(3):
        mean, cov = kf.predict(mean, cov)
    assert mean[0, 3] < 0
    assert np.all(np.isfinite(cov)) and np.all(np.linalg.eigvalsh(cov) > 0)
