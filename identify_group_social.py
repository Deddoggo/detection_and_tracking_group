import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import cv2

def get_colors(n):
    if n <= 0: return []
    colors = []
    hue_partition = 179 // n  
    for i in range(n):
        hue = int(i * hue_partition)
        hsv_color = np.uint8([[[hue, 240, 220]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, rgb_color[::-1])))
    return colors

def cluster_bboxes_with_ids(groups_status, bboxes, track_ids, max_group_id, 
                            pose_vectors=None, # <--- THÊM THAM SỐ NÀY
                            homography_matrix=None, 
                            scale_factor=1.0,
                            eps=1.5, 
                            min_samples=2, 
                            threshold_overlap=0.5):
    
    if not bboxes or not track_ids or len(bboxes) != len(track_ids):
        return [], groups_status, max_group_id
    
    N = len(bboxes)
    
    # 1. TÍNH ĐIỂM CHẠM CHÂN VÀ BIẾN ĐỔI TỌA ĐỘ
    points_to_cluster = []
    for (x, y, w, h) in bboxes:
        points_to_cluster.append([x + w / 2, y + h])
        
    real_world_coords = [None] * N
    if homography_matrix is not None and N > 0:
        pts_src = np.array([points_to_cluster], dtype=np.float32)
        pts_dst = cv2.perspectiveTransform(pts_src, homography_matrix)
        X = pts_dst[0] * scale_factor
        real_world_coords = X
    else:
        X = np.array(points_to_cluster)

    # 2. ÁNH XẠ POSE VECTOR XUỐNG MẶT ĐẤT (BEV)
    bev_poses = [None] * N
    if homography_matrix is not None and pose_vectors is not None:
        for i in range(N):
            if pose_vectors[i] is not None:
                foot_x, foot_y = points_to_cluster[i]
                vx, vy = pose_vectors[i]
                
                # Chiếu 2 điểm: Gót chân và Mũi tên hướng
                pt1 = np.array([[[foot_x, foot_y]]], dtype=np.float32)
                pt2 = np.array([[[foot_x + vx*10, foot_y + vy*10]]], dtype=np.float32)
                
                bev_pt1 = cv2.perspectiveTransform(pt1, homography_matrix)[0][0]
                bev_pt2 = cv2.perspectiveTransform(pt2, homography_matrix)[0][0]
                
                bev_v = bev_pt2 - bev_pt1
                norm = np.linalg.norm(bev_v)
                if norm > 1e-6:
                    bev_poses[i] = bev_v / norm
            
    # 3. XÂY DỰNG MA TRẬN KHOẢNG CÁCH KẾT HỢP HƯỚNG NHÌN
    # Thay np.inf bằng 1000.0 để lách lỗi của sklearn. 
    # Mức này đủ lớn để DBSCAN từ chối gom nhóm (vì nó lớn hơn hẳn eps).
    dist_matrix = np.full((N, N), 1000.0)
    
    for i in range(N):
        dist_matrix[i, i] = 0.0
        for j in range(i + 1, N):
            # A. Kiểm tra khoảng cách vật lý
            dist = np.linalg.norm(X[i] - X[j])
            if dist > eps:
                continue # Đứng quá xa -> Bỏ qua
                
            # B. Kiểm tra tương tác xã hội (Orientation)
            connected = False
            pi, pj = bev_poses[i], bev_poses[j]
            
            if pi is None or pj is None:
                connected = True # Fallback: Thiếu pose thì chỉ tin vào khoảng cách
            else:
                # Vector nối từ người i đến người j
                v_ij = X[j] - X[i]
                norm_ij = np.linalg.norm(v_ij)
                
                if norm_ij < 1e-6:
                    connected = True
                else:
                    v_ij_dir = v_ij / norm_ij
                    v_ji_dir = -v_ij_dir
                    
                    # Tính góc nhìn (Cosine Similarity)
                    # Góc > 0 nghĩa là lệch nhau < 90 độ (đang hướng về phía nhau)
                    cos_i_to_j = np.dot(pi, v_ij_dir) 
                    cos_j_to_i = np.dot(pj, v_ji_dir)
                    cos_i_j = np.dot(pi, pj) # Xem có đi song song không
                    
                    # Tiêu chí F-Formation:
                    # 1. Đi song song (cùng hướng): Góc giữa 2 pose < 45 độ (cos > 0.7)
                    side_by_side = (cos_i_j > 0.7)
                    # 2. Mặt đối mặt / L-shape: Một trong hai người hướng về phía người kia
                    interacting = (cos_i_to_j > 0.3) or (cos_j_to_i > 0.3)
                    
                    if side_by_side or interacting:
                        connected = True

            # Chỉ kết nối nếu thỏa mãn cả Khoảng cách VÀ Góc nhìn
            if connected:
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    # 4. CHẠY DBSCAN VỚI MA TRẬN TÙY CHỈNH
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(dist_matrix)

    # --- 5. LOGIC DUY TRÌ ID NHÓM (GIỮ NGUYÊN) ---
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
    for i, (bbox, id_p, label) in enumerate(zip(bboxes, track_ids, labels)):
        id_g = idp_to_idg_map.get(id_p, -1)
        color = (128, 128, 128) if id_g == -1 else color_map.get(id_g, (255,255,255))
        real_pos = real_world_coords[i] if i < len(real_world_coords) else None

        results.append({
            'id_p': id_p,
            'bbox': bbox,
            'id_g': id_g,
            'color': color,
            'real_pos': real_pos 
        })
        
    return results, new_group_status, max_group_id