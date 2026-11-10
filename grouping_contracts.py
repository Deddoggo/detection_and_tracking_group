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


def geometry_contract(mode, H=None, *, calibration=None, video=None, close_distance=None):
    """Metric validity is an explicit supplied assertion with supporting provenance.

    This validates evidence structure/identity, not the truth of field measurements.
    Legacy mode deliberately preserves the old unverified-unit behavior.
    """
    if mode not in ('metric', 'image-space-diagnostic', 'legacy-b0'):
        raise ValueError('explicit geometry mode required')
    evidence_level = None
    numerical_fit = None
    if mode == 'metric':
        from observation_cache import canonical_hash
        if H is None or calibration is None or video is None:
            raise ValueError('metric grouping requires verified calibration provenance and video identity')
        _,_,H = validate_grouping_inputs([],[],homography_matrix=H)
        user_confirmed = calibration.get('verification_level') == 'user-confirmed'
        if (calibration.get('schema') != 'calibration.v1' or calibration.get('units') != 'm'
                or not calibration.get('calibration_id') or not calibration.get('measurement_evidence')):
            raise ValueError('unverified metric calibration')
        if user_confirmed:
            confirmation = calibration.get('measurement_confirmation', {})
            if (not isinstance(confirmation,dict) or calibration.get('verified') is not False
                    or confirmation.get('correspondence_relationship_confirmed') is not True
                    or not all(confirmation.get(k) for k in ('confirmed_by','confirmed_on','source','source_sha256'))
                    or calibration.get('independently_surveyed') is not False
                    or calibration.get('physical_projection_error_measured') is not False
                    or calibration.get('physical_projection_error_m') is not None
                    or calibration.get('correspondence_tolerance_m') is not None):
                raise ValueError('user-confirmed scale must not claim independent physical verification/error bounds')
            evidence_level = 'user-confirmed'
        else:
            # Backward-compatible assertion contract. This flag alone is not a
            # certificate of independent survey or measured projection accuracy.
            if calibration.get('verified') is not True or not calibration.get('verified_by'):
                raise ValueError('unverified metric calibration')
            evidence_level = 'legacy-verified-assertion'
        for key in ('video_sha256', 'width', 'height'):
            if calibration.get(key) != video.get(key):
                raise ValueError(f'calibration/video identity mismatch: {key}')
        if calibration.get('homography_sha256') != canonical_hash(np.asarray(H).tolist()):
            raise ValueError('calibration homography hash mismatch')
        points = calibration.get('correspondences', [])
        if not isinstance(points,list) or len(points) < 4 or any(not isinstance(p,dict) or
                np.asarray(p.get('image_xy')).shape != (2,) or np.asarray(p.get('ground_xy')).shape != (2,)
                for p in points):
            raise ValueError('at least four finite image/ground correspondences required')
        image_points = np.asarray([p['image_xy'] for p in points],dtype=float)
        ground_points = np.asarray([p['ground_xy'] for p in points],dtype=float)
        if not np.isfinite(image_points).all() or not np.isfinite(ground_points).all():
            raise ValueError('nonfinite calibration correspondence')
        if user_confirmed:
            dimensions = np.asarray(confirmation.get('dimensions_m'),dtype=float)
            if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
                raise ValueError('confirmed positive rectangle dimensions required')
            w,h = dimensions
            if not np.array_equal(ground_points,[[0,0],[w,0],[w,h],[0,h]]):
                raise ValueError('ground correspondences differ from confirmed rectangle/order')
            precision = calibration.get('correspondence_numeric_dtype')
            if precision not in ('float32','float64'):
                raise ValueError('documented construction precision required for numerical fit check')
            # A representation-precision check only: source construction explicitly
            # used float32 targets. This is NOT a physical projection error bound.
            tolerance = float(np.finfo(precision).eps * max(1.,np.linalg.norm(ground_points,axis=1).max()))
        else:
            tolerance = calibration.get('correspondence_tolerance_m')
            if tolerance is None or not np.isfinite(tolerance) or tolerance <= 0:
                raise ValueError('explicit positive correspondence_tolerance_m required')
        check_projection_points(image_points,H)
        homogeneous = np.column_stack([image_points,np.ones(len(image_points))]) @ H.T
        projected = homogeneous[:,:2]/homogeneous[:,2:]
        residual = float(np.max(np.linalg.norm(projected-ground_points,axis=1)))
        if residual > tolerance:
            raise ValueError('homography does not reproduce supplied ground correspondences')
        numerical_fit = {'max_residual_declared_units':residual,'numeric_limit_declared_units':tolerance,
                         'basis':'construction floating-point precision' if user_confirmed else 'supplied legacy fit tolerance',
                         'physical_error_bound':False}
        units = 'm'
    elif mode == 'image-space-diagnostic':
        if H is not None or close_distance is None:
            raise ValueError('image diagnostic requires no homography and explicit pixel close_distance')
        units = 'pixels'
    else:
        units = 'unverified-legacy'
    close_distance = .5 if close_distance is None else close_distance
    if not np.isfinite(close_distance) or close_distance <= 0:
        raise ValueError('close_distance must be finite and positive')
    result = {'mode': mode, 'units': units, 'close_distance': float(close_distance)}
    if mode == 'metric':
        result.update(calibration_evidence_level=evidence_level,numerical_fit=numerical_fit,
                      independent_physical_accuracy_verified=False)
    return result


def project_geometry(bboxes, pose_vectors, H, scale_factor=1.0, *, image_directions=False):
    """The original float32 footpoint/10-pixel direction projection, shared with replay."""
    import cv2
    points = [[x+w/2, y+h] for x,y,w,h in bboxes]
    poses = [None] * len(points)
    if not points:
        return np.empty((0,2)), poses
    if H is None:
        if image_directions and pose_vectors is not None:
            poses = list(pose_vectors)
        return np.array(points), poses
    src = np.array([points], dtype=np.float32)
    check_projection_points(src,H)
    X = cv2.perspectiveTransform(src,H)[0] * scale_factor
    require_finite_projection(X)
    if pose_vectors is not None:
        for i, direction in enumerate(pose_vectors):
            if direction is None:
                continue
            foot_x, foot_y = points[i]
            vx, vy = direction
            p1 = np.array([[[foot_x,foot_y]]],dtype=np.float32)
            p2 = np.array([[[foot_x+vx*10,foot_y+vy*10]]],dtype=np.float32)
            check_projection_points(p1,H)
            check_projection_points(p2,H)
            delta = cv2.perspectiveTransform(p2,H)[0][0] - cv2.perspectiveTransform(p1,H)[0][0]
            require_finite_projection(delta)
            norm = np.linalg.norm(delta)
            if norm > 1e-6:
                poses[i] = delta/norm
    return X, poses
