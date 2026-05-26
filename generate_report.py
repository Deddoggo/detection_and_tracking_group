import json
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
import math
import os

INPUT_JSON = "playground_results/final_sociometrics_data.json"
OUTPUT_EXCEL = "playground_results/Sociometrics_Report.xlsx"
FPS = 30.0
INTERVAL_SEC = 30
FRAMES_PER_INTERVAL = int(FPS * INTERVAL_SEC)

print("Đang tải JSON data...")
with open(INPUT_JSON, 'r') as f:
    raw_data = json.load(f)

# 1. Flatten JSON vào list để tạo DataFrame
records = []
for frame_key, persons in raw_data.items():
    frame_idx = int(frame_key.split('_')[1])
    interval_id = frame_idx // FRAMES_PER_INTERVAL
    
    for p in persons:
        if p.get('real_pos') is not None:
            records.append({
                'frame': frame_idx,
                'interval_id': interval_id,
                'id_p': p['id_p'],
                'id_g': p['id_g'],
                'gender': str(p.get('gender', 'Unknown')).capitalize(),
                'x': p['real_pos'][0],
                'y': p['real_pos'][1]
            })

df = pd.DataFrame(records)

print(f"Bắt đầu phân tích {len(df)} điểm dữ liệu thực tế...")

# ==========================================
# METRICS 1: DYNAMICS & DISTANCES (30s)
# ==========================================
interval_stats = []
intervals = df['interval_id'].unique()

def get_clean_area(pts):
    """Tính Convex Hull sau khi loại bỏ 5% điểm xa tâm nhất (outliers)"""
    if len(pts) < 3: return 0.0
    centroid = np.mean(pts, axis=0)
    distances = np.linalg.norm(pts - centroid, axis=1)
    p95 = np.percentile(distances, 95)
    clean_pts = pts[distances <= p95]
    if len(clean_pts) < 3: return 0.0
    return ConvexHull(clean_pts).volume

for iv in sorted(intervals):
    df_iv = df[df['interval_id'] == iv]
    start_sec = iv * INTERVAL_SEC
    end_sec = start_sec + INTERVAL_SEC
    
    # Số nhóm hình thành (bỏ qua id_g = -1)
    groups_formed = df_iv[df_iv['id_g'] != -1]['id_g'].nunique()
    
    # Diện tích chiếm dụng (m2) bằng Convex Hull đã được LỌC NHIỄU
    pts_F = df_iv[df_iv['gender'] == 'Female'][['x', 'y']].to_numpy()
    pts_M = df_iv[df_iv['gender'] == 'Male'][['x', 'y']].to_numpy()
    
    space_F = get_clean_area(pts_F)
    space_M = get_clean_area(pts_M)
    
    # Khoảng cách gần nhất trung bình (F-M, F-F, M-M, M-F)
    dist_FF_list, dist_FM_list, dist_MM_list, dist_MF_list = [], [], [], []
    
    for frame, group in df_iv.groupby('frame'):
        females = group[group['gender'] == 'Female'][['x', 'y']].values
        males = group[group['gender'] == 'Male'][['x', 'y']].values
        
        if len(females) > 1:
            dists = distance_matrix(females, females)
            np.fill_diagonal(dists, np.inf)
            dist_FF_list.extend(dists.min(axis=1))
        
        if len(males) > 1:
            dists = distance_matrix(males, males)
            np.fill_diagonal(dists, np.inf)
            dist_MM_list.extend(dists.min(axis=1))
            
        if len(females) > 0 and len(males) > 0:
            dists_fm = distance_matrix(females, males)
            dist_FM_list.extend(dists_fm.min(axis=1)) # Nữ tìm Nam gần nhất
            dist_MF_list.extend(dists_fm.min(axis=0)) # Nam tìm Nữ gần nhất

    # Tính tổng quãng đường di chuyển (Travel Distances)
    travel_dist = {'Female': 0.0, 'Male': 0.0}
    for id_p, p_data in df_iv.groupby('id_p'):
        p_data = p_data.sort_values('frame')
        coords = p_data[['x', 'y']].values
        if len(coords) > 1:
            # Tính khoảng cách dịch chuyển giữa các frame liên tiếp
            dists = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            gender = p_data['gender'].iloc[0]
            if gender in travel_dist:
                travel_dist[gender] += np.sum(dists)

    interval_stats.append({
        'Interval': f"{start_sec}s - {end_sec}s",
        'Groups Formed': groups_formed,
        'Space Occupied Girls (m2)': round(space_F, 2),
        'Space Occupied Boys (m2)': round(space_M, 2),
        'Travel Dist Girls (m)': round(travel_dist['Female'], 2),
        'Travel Dist Boys (m)': round(travel_dist['Male'], 2),
        'Avg Dist F to F (m)': round(np.mean(dist_FF_list) if dist_FF_list else 0, 2),
        'Var Dist F to F': round(np.var(dist_FF_list) if dist_FF_list else 0, 2),
        'Avg Dist F to M (m)': round(np.mean(dist_FM_list) if dist_FM_list else 0, 2),
        'Var Dist F to M': round(np.var(dist_FM_list) if dist_FM_list else 0, 2),
        'Avg Dist M to M (m)': round(np.mean(dist_MM_list) if dist_MM_list else 0, 2),
        'Var Dist M to M': round(np.var(dist_MM_list) if dist_MM_list else 0, 2),
    })

df_intervals = pd.DataFrame(interval_stats)

# ==========================================
# METRICS 2: GROUP DURATION & COMPOSITION (FIXED)
# ==========================================
group_stats = []
df_valid_groups = df[df['id_g'] != -1]

# Ngưỡng đứt gãy: Nếu nhóm biến mất quá 2 giây (60 frames tại 30FPS), 
# coi như nhóm đã rã. Lần xuất hiện tiếp theo sẽ tính là một "Session" mới.
MAX_GAP_FRAMES = int(FPS * 2.0) 

for id_g, g_data in df_valid_groups.groupby('id_g'):
    # Lấy danh sách các frame mà ID nhóm này xuất hiện, sắp xếp tăng dần
    unique_frames = sorted(g_data['frame'].unique())
    
    # Tách các frame thành các Sessions liên tục
    sessions = []
    current_session = [unique_frames[0]]
    
    for i in range(1, len(unique_frames)):
        # Nếu khoảng cách giữa 2 frame lớn hơn ngưỡng -> Ngắt session
        if unique_frames[i] - unique_frames[i-1] > MAX_GAP_FRAMES:
            sessions.append(current_session)
            current_session = [unique_frames[i]]
        else:
            current_session.append(unique_frames[i])
    sessions.append(current_session) # Add session cuối cùng
    
    # Tính toán Metrics cho từng Session thực tế
    for session_idx, session_frames in enumerate(sessions):
        first_f = session_frames[0]
        last_f = session_frames[-1]
        
        duration_sec = (last_f - first_f) / FPS
        
        # Nếu session chỉ diễn ra trong 1 frame (cùng f_min và f_max)
        # Bù thêm thời gian của FRAME_STRIDE (mặc định 3 frame ~ 0.1s)
        if duration_sec == 0:
            duration_sec = 3 / FPS 
            
        # Lọc nhiễu: Bỏ qua các nhóm tồn tại chớp nhoáng dưới 1 giây
        if duration_sec < 1.0:
            continue
            
        # Lọc data chỉ thuộc về session hiện tại
        session_data = g_data[g_data['frame'].isin(session_frames)]
        
        unique_members = session_data.drop_duplicates(subset=['id_p'])
        f_count = sum(unique_members['gender'] == 'Female')
        m_count = sum(unique_members['gender'] == 'Male')
        size = f_count + m_count
        
        group_stats.append({
            'Group ID': f"{id_g} (Session {session_idx + 1})", # Tách ID theo lần tụ tập
            'Duration (seconds)': round(duration_sec, 1),
            'Total Size': size,
            'Female Count': f_count,
            'Male Count': m_count,
            'First Frame': first_f,
            'Last Frame': last_f
        })

df_groups = pd.DataFrame(group_stats)

# ==========================================
# EXPORT TO EXCEL
# ==========================================
print(f"Đang lưu báo cáo ra Excel tại: {OUTPUT_EXCEL}")
with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
    df_intervals.to_excel(writer, sheet_name='Interval Data (30s)', index=False)
    df_groups.to_excel(writer, sheet_name='Group Dynamics', index=False)
    
    # Sheet Meta-data để giải thích cho đối tác
    meta_df = pd.DataFrame([
        {'Variable': 'Group ID', 'Explanation': 'Unique identifier for a social cluster. ID > 0 implies interaction. ID = -1 means the person is isolated.'},
        {'Variable': 'Space Occupied (m2)', 'Explanation': 'Calculated using Convex Hull algorithm over the real-world coordinates.'},
        {'Variable': 'Var Dist', 'Explanation': 'Variance of the distance, representing the extent of variation over the period.'}
    ])
    meta_df.to_excel(writer, sheet_name='Metadata & Clarifications', index=False)

print("✅ Hoàn thành xuất Report Analytics.")