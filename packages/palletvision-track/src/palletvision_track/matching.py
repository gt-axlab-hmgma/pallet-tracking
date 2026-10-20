"""Match predicted track boxes to detections: an IoU cost matrix and a thresholded assignment.

ByteTrack runs this twice per frame (high-score detections, then low-score ones) with different
thresholds, so the thresholds are arguments here and nothing is hard-coded.
"""

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from palletvision_track.boxes import Array, pairwise_iou

IntArray = npt.NDArray[np.intp]


def iou_distance(
    track_boxes: npt.ArrayLike,
    det_boxes: npt.ArrayLike,
    det_scores: npt.ArrayLike | None = None,
) -> Array:
    """Cost of pairing track ``i`` with detection ``j``: ``1 - IoU``, shape ``(N, M)``.

    Boxes are corners. 0 means a perfect overlap and 1 means none. With ``det_scores`` the IoU is
    first multiplied by the detection's score (score fusion), so of two equally overlapping
    detections the more confident one is the cheaper match.
    """
    iou = pairwise_iou(track_boxes, det_boxes)
    if det_scores is not None:
        iou = iou * np.asarray(det_scores, dtype=np.float64)[None, :]
    return 1.0 - iou


def assign(cost: Array, thresh: float) -> tuple[IntArray, IntArray, IntArray]:
    """One-to-one matching of rows to columns that leaves too-costly pairs unmatched.

    Returns ``(matches, unmatched_rows, unmatched_cols)``: ``matches`` is ``(K, 2)`` of
    ``(row, col)``, the other two are index arrays.

    Leaving a row and a column unmatched is allowed at a fixed price of ``thresh`` for the pair
    (``thresh / 2`` each). The matching minimises matched costs plus those prices, so a real pair
    is used only if it costs less than ``thresh``, and one very good pair can beat two
    mediocre ones. This is the objective of ``lap.lapjv(cost, extend_cost=True,
    cost_limit=thresh)``, which the official ByteTrack uses. A pair costing exactly ``thresh`` is
    a tie with leaving both unmatched.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), dtype=np.intp), np.arange(n), np.arange(m)
    # Square (n + m) matrix. Rows: n tracks, then one dummy row per detection. Columns: m
    # detections, then one dummy column per track. A dummy standing in for "unmatched" costs
    # thresh / 2; two dummies facing each other cost nothing, so they can always pair off.
    extended = np.full((n + m, n + m), thresh / 2)
    extended[n:, m:] = 0.0
    extended[:n, :m] = cost
    rows, cols = linear_sum_assignment(extended)
    real = (rows < n) & (cols < m)  # a real track on a real detection, not on a dummy
    rows, cols = rows[real], cols[real]
    row_free = np.ones(n, dtype=bool)
    col_free = np.ones(m, dtype=bool)
    row_free[rows] = False
    col_free[cols] = False
    return np.stack([rows, cols], axis=1), np.flatnonzero(row_free), np.flatnonzero(col_free)
