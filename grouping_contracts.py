"""Small contracts shared by the two grouping entry points; no model imports."""
from numbers import Integral
import numpy as np


def validate_grouping_inputs(bboxes, track_ids, pose_vectors=None,
                             homography_matrix=None, scale_factor=1.0,
                             eps=1.5, min_samples=2):
    boxes = np.asarray(bboxes, dtype=float)
    if boxes.shape == (0,):
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("bboxes must be a finite (N, 4) xywh collection")
    if np.any(boxes[:, 2:] <= 0):
        raise ValueError("bbox width and height must be positive")
    try:
        ids = list(track_ids)
    except TypeError as exc:
        raise ValueError("track_ids must be a one-dimensional ID collection") from exc
    if len(ids) != len(boxes):
        raise ValueError("bboxes and track_ids must have equal lengths")
    if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in ids):
        raise ValueError("track_ids must be nonnegative integers")
    ids = [int(i) for i in ids]
    if len(ids) != len(set(ids)):
        raise ValueError("track_ids must be unique within a frame")
    if pose_vectors is not None:
        try:
            count = len(pose_vectors)
        except TypeError as exc:
            raise ValueError("pose_vectors must be a direction collection or None") from exc
        if count != len(boxes):
            raise ValueError("pose_vectors and bboxes must have equal lengths")
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be finite and positive")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if isinstance(min_samples, bool) or not isinstance(min_samples, Integral) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    H = None
    if homography_matrix is not None:
        H = np.asarray(homography_matrix)
        if not np.issubdtype(H.dtype, np.floating):
            H = H.astype(float)
        if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H) != 3:
            raise ValueError("homography_matrix must be finite, nonsingular and (3, 3)")
    return boxes.tolist(), ids, H


def check_projection_points(points, H):
    """Reject a projective horizon before OpenCV may map it to a false origin."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    if not np.isfinite(homogeneous).all() or np.any(homogeneous[:, 2] == 0):
        raise ValueError("points have a nonfinite projection or zero homogeneous denominator")


def require_finite_projection(points):
    if not np.isfinite(points).all():
        raise ValueError("projected coordinates must be finite")
