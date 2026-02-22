import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

import random
import cv2

def get_colors(n):
    """
    Generate n visually distinct colors using HSV color space
    
    Args:
        n: Number of distinct colors needed
    Returns:
        List of RGB tuples with n distinct colors. Returns empty list if n <= 0.
    """
    if n <= 0:
        return []
        
    colors = []
    hue_partition = 179 // n  
    
    for i in range(n):
        # Create color with:
        # - Evenly spaced hue for maximum distinction
        # - High saturation (240) for vivid colors
        # - High value (220) for better visibility
        hue = int(i * hue_partition)
        saturation = 240
        value = 220
        
        hsv_color = np.uint8([[[hue, saturation, value]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, rgb_color[::-1])))
    
    return colors

def cluster_bboxes_with_ids(groups_status, bboxes, track_ids, max_group_id, 
                            homography_matrix=None, 
                            scale_factor=1.0,
                            eps=1.5, 
                            min_samples=2, 
                            threshold_overlap=0.5):
    
    if not bboxes or not track_ids or len(bboxes) != len(track_ids):
        return [], groups_status, max_group_id
    
    # 1. TÍNH ĐIỂM CHẠM CHÂN (Foot Points)
    points_to_cluster = []
    for (x, y, w, h) in bboxes:
        foot_x = x + w / 2
        foot_y = y + h 
        points_to_cluster.append([foot_x, foot_y])

    # 2. BIẾN ĐỔI TỌA ĐỘ
    real_world_coords = [] # Lưu tọa độ thực để trả về
    
    if homography_matrix is not None and len(points_to_cluster) > 0:
        pts_src = np.array([points_to_cluster], dtype=np.float32)
        pts_dst = cv2.perspectiveTransform(pts_src, homography_matrix)
        X = pts_dst[0] * scale_factor
        real_world_coords = X # X chính là tọa độ (x_mét, y_mét)
    else:
        X = np.array(points_to_cluster)
        real_world_coords = [None] * len(X) # Không có matrix thì không có tọa độ mét

    # 3. CHẠY DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    # ... (Phần logic Grouping giữ nguyên) ...
    cluster_members = defaultdict(list)
    for label, tid in zip(labels, track_ids):
        if label != -1:
            cluster_members[label].append(tid)

    new_group_status = defaultdict(list)
    if len(groups_status) > 0:       
        for id_cluster, members in cluster_members.items():
            best_match = None
            best_overlap = 0
            overlap_ratio = 0
            for group_id, old_group_members in groups_status.items():
                overlap = len(set(members) & set(old_group_members))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = group_id
                    overlap_ratio = best_overlap / len(old_group_members)

            if best_match is not None and overlap_ratio >= threshold_overlap:
                new_group_status[best_match] = cluster_members[id_cluster]
            else:
                new_group_status[max_group_id + 1] = cluster_members[id_cluster]
                max_group_id += 1
    else:
        new_group_status = cluster_members
        if new_group_status:
             max_group_id = max(new_group_status.keys(), default=max_group_id)

    n_clusters = len(new_group_status)
    colors = get_colors(n_clusters)
    color_map = dict(zip(new_group_status.keys(), colors))
    idp_to_idg_map = {id_p: id_g for id_g, id_ps in new_group_status.items() for id_p in id_ps}

    results = []
    # Thêm i để lấy tọa độ thực tương ứng từ real_world_coords
    for i, (bbox, id_p, id_g) in enumerate(zip(bboxes, track_ids, labels)):
        id_g = idp_to_idg_map.get(id_p, -1)
        color = (128, 128, 128) if id_g == -1 else color_map.get(id_g, (255,255,255))
        
        # Lấy tọa độ thực của người thứ i
        real_pos = real_world_coords[i] if i < len(real_world_coords) else None

        results.append({
            'id_p': id_p,
            'bbox': bbox,
            'id_g': id_g,
            'color': color,
            'real_pos': real_pos # <--- TRẢ VỀ THÊM CÁI NÀY
        })
        
    return results, new_group_status, max_group_id