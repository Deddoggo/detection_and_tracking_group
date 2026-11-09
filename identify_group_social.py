import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import cv2
from group_identity import associate_groups
from grouping_contracts import (validate_grouping_inputs, check_projection_points,
                                require_finite_projection, geometry_contract, project_geometry)
from orientation_gate import MODES, heading_or_none, pair_compatibility

def get_colors(n):
    if n <= 0:
        return []

    colors = []
    hue_partition = 179 // n
    for i in range(n):
        hue = int(i * hue_partition)
        hsv_color = np.uint8([[[hue, 240, 220]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, rgb_color[::-1])))
    return colors


def cluster_bboxes_with_ids(groups_status, bboxes, track_ids, max_group_id,
                            pose_vectors=None,
                            homography_matrix=None,
                            scale_factor=1.0,
                            eps=1.5,
                            min_samples=2,
                            threshold_overlap=0.5, *, orientation_mode="B0-original",
                            geometry_mode="metric", calibration=None, video_identity=None,
                            close_distance=None):

    bboxes, track_ids, homography_matrix = validate_grouping_inputs(
        bboxes, track_ids, pose_vectors, homography_matrix, scale_factor, eps, min_samples)
    geometry = geometry_contract(geometry_mode, homography_matrix, calibration=calibration,
                                 video=video_identity, close_distance=close_distance)
    if geometry_mode == "metric" and scale_factor != 1.0:
        raise ValueError("metric scale belongs in the calibrated homography; scale_factor must be 1")
    if orientation_mode not in MODES:
        raise ValueError(f"orientation_mode must be one of {MODES}")
    if eps >= 1000:
        raise ValueError("social eps must be below the B0 rejected-distance sentinel 1000")
    if pose_vectors is not None:
        pose_vectors = [heading_or_none(v) for v in pose_vectors]
    if not bboxes:
        _, state, maximum = associate_groups(groups_status, {}, max_group_id, threshold_overlap)
        return [], state, maximum

    N = len(bboxes)

    X, bev_poses = project_geometry(bboxes, pose_vectors, homography_matrix, scale_factor,
                                   image_directions=geometry_mode == 'image-space-diagnostic')
    real_world_coords = X if homography_matrix is not None else [None] * N

    # 3) Build custom pairwise matrix combining distance and social orientation cues.
    dist_matrix = np.full((N, N), 1000.0)

    for i in range(N):
        dist_matrix[i, i] = 0.0
        for j in range(i + 1, N):
            # A) Physical proximity gate.
            dist = np.linalg.norm(X[i] - X[j])
            if dist > eps:
                continue

            # B0 is still the default. Alternative modes are diagnostic controls.
            connected = pair_compatibility(
                bev_poses[i], bev_poses[j], X[j] - X[i], eps,
                mode=orientation_mode, close_distance=geometry["close_distance"])["compatible"]

            if connected:
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    # 4) Cluster from the custom precomputed matrix.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(dist_matrix)

    # 5) Keep group IDs stable across frames by overlap with previous members.
    cluster_members = defaultdict(list)
    for label, tid in zip(labels, track_ids):
        if label != -1:
            cluster_members[label].append(tid)

    _, new_group_status, max_group_id = associate_groups(
        groups_status, cluster_members, max_group_id, threshold_overlap)

    n_clusters = len(new_group_status)
    colors = get_colors(n_clusters)
    color_map = dict(zip(new_group_status.keys(), colors))
    idp_to_idg_map = {id_p: id_g for id_g, id_ps in new_group_status.items() for id_p in id_ps}

    results = []
    for i, (bbox, id_p, _label) in enumerate(zip(bboxes, track_ids, labels)):
        id_g = idp_to_idg_map.get(id_p, -1)
        color = (128, 128, 128) if id_g == -1 else color_map.get(id_g, (255, 255, 255))
        real_pos = real_world_coords[i] if i < len(real_world_coords) else None

        results.append({
            'id_p': id_p,
            'bbox': bbox,
            'id_g': id_g,
            'color': color,
            'real_pos': real_pos
        })

    return results, new_group_status, max_group_id
