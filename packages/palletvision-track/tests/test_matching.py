import itertools

import numpy as np
import pytest

from palletvision_track.matching import assign, iou_distance


def pairs(matches: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in matches}


def test_iou_distance_identical_is_zero_and_disjoint_is_one() -> None:
    tracks = [[0, 0, 10, 10], [50, 50, 60, 60]]
    dets = [[0, 0, 10, 10], [100, 100, 110, 110]]
    cost = iou_distance(tracks, dets)
    assert cost.shape == (2, 2)
    assert cost[0, 0] == 0
    np.testing.assert_array_equal(cost[[0, 1, 1], [1, 0, 1]], 1)


def test_iou_distance_score_fusion_prefers_the_confident_detection() -> None:
    same_box = [[0, 0, 10, 10], [0, 0, 10, 10]]
    cost = iou_distance([[0, 0, 10, 10]], same_box, det_scores=[0.9, 0.5])
    np.testing.assert_allclose(cost, [[1 - 0.9, 1 - 0.5]])


def test_iou_distance_empty() -> None:
    assert iou_distance(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)
    assert iou_distance([[0, 0, 1, 1]], np.zeros((0, 4))).shape == (1, 0)


def test_assign_known_3x3() -> None:
    cost = np.array([[0.1, 0.9, 0.8], [0.7, 0.2, 0.9], [0.8, 0.6, 0.3]])
    matches, rows, cols = assign(cost, 0.5)
    assert pairs(matches) == {(0, 0), (1, 1), (2, 2)}
    assert len(rows) == len(cols) == 0


def test_assign_chooses_the_global_optimum_not_the_greedy_pairing() -> None:
    # greedy would take the 0.1 at (0, 0) and force (1, 1) at 0.9; the optimum is 0.2 + 0.3
    cost = np.array([[0.1, 0.2], [0.3, 0.9]])
    matches, _, _ = assign(cost, 1.0)
    assert pairs(matches) == {(0, 1), (1, 0)}


def test_assign_rejects_pairs_over_the_threshold() -> None:
    cost = np.array([[0.2, 0.9], [0.9, 0.7]])
    matches, rows, cols = assign(cost, 0.5)
    assert pairs(matches) == {(0, 0)}
    assert list(rows) == [1]
    assert list(cols) == [1]


def test_assign_matches_a_pair_just_under_the_threshold_but_not_just_over() -> None:
    assert pairs(assign(np.array([[0.49]]), 0.5)[0]) == {(0, 0)}
    matches, rows, cols = assign(np.array([[0.51]]), 0.5)
    assert len(matches) == 0
    assert list(rows) == [0] and list(cols) == [0]


def test_assign_prefers_one_excellent_pair_to_two_mediocre_ones() -> None:
    # Pairing {(0, 0), (1, 1)} costs 0.5 + 0.5 = 1.0. Matching only (0, 1) at 0.0 and leaving
    # track 1 and detection 0 unmatched costs 0.0 + 2 * (0.5 / 2) = 0.5, which is cheaper.
    cost = np.array([[0.5, 0.0], [0.9, 0.5]])
    matches, rows, cols = assign(cost, 0.5)
    assert pairs(matches) == {(0, 1)}
    assert list(rows) == [1] and list(cols) == [0]


def test_assign_rectangular_leaves_the_extra_ones_unmatched() -> None:
    cost = np.array([[0.1, 0.8, 0.8], [0.8, 0.1, 0.8]])
    matches, rows, cols = assign(cost, 0.5)
    assert pairs(matches) == {(0, 0), (1, 1)}
    assert len(rows) == 0
    assert list(cols) == [2]


@pytest.mark.parametrize(("n", "m"), [(0, 0), (0, 3), (3, 0)])
def test_assign_empty(n: int, m: int) -> None:
    matches, rows, cols = assign(np.zeros((n, m)), 0.5)
    assert matches.shape == (0, 2)
    assert list(rows) == list(range(n))
    assert list(cols) == list(range(m))


def best_objective_by_brute_force(cost: np.ndarray, thresh: float) -> float:
    """Lowest (sum of matched costs + thresh / 2 per unmatched row or column) over all matchings."""
    n, m = cost.shape
    best: float = (n + m) * thresh / 2  # nothing matched
    for k in range(1, min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                total = sum(cost[r, c] for r, c in zip(rows, cols, strict=True))
                best = min(best, float(total) + (n + m - 2 * k) * thresh / 2)
    return best


def test_assign_reaches_the_best_objective_on_random_matrices() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, m = rng.integers(1, 5, size=2)
        cost = rng.random((n, m))
        matches, rows, cols = assign(cost, 0.6)
        k = len(matches)
        got = cost[matches[:, 0], matches[:, 1]].sum() + (n + m - 2 * k) * 0.6 / 2
        assert got == pytest.approx(best_objective_by_brute_force(cost, 0.6))
        assert np.all(cost[matches[:, 0], matches[:, 1]] <= 0.6)  # never a too-costly pair
        assert len(rows) == n - k and len(cols) == m - k
        assert len(set(matches[:, 0])) == k == len(set(matches[:, 1]))
