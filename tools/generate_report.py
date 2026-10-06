# Allow both python -m PACKAGE.NAME and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import repo_path, checkpoint_path, external_path, output_path, load_paths
import sys as _dependency_sys
_dependency_sys.path.insert(0, external_path())
import json
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
import math
import os

INPUT_JSON = output_path('final_sociometrics_data.json')
OUTPUT_EXCEL = output_path('Sociometrics_Report.xlsx')
FPS = 30.0
INTERVAL_SEC = 30
FRAMES_PER_INTERVAL = int(FPS * INTERVAL_SEC)

print("Đang tải JSON data...")
with open(INPUT_JSON, 'r') as f:
    raw_data = json.load(f)

# 1. Flatten JSON into a list to create a DataFrame.
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
    """Compute the convex hull after removing the 5% of points farthest from the center (outliers)."""
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
    
    # Number of formed groups (exclude id_g = -1).
    groups_formed = df_iv[df_iv['id_g'] != -1]['id_g'].nunique()
    
    # Occupied area (m2) from a NOISE-FILTERED convex hull.
    pts_F = df_iv[df_iv['gender'] == 'Female'][['x', 'y']].to_numpy()
    pts_M = df_iv[df_iv['gender'] == 'Male'][['x', 'y']].to_numpy()
    
    space_F = get_clean_area(pts_F)
    space_M = get_clean_area(pts_M)
    
    # Mean nearest-neighbor distances (F-M, F-F, M-M, M-F).
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
            dist_FM_list.extend(dists_fm.min(axis=1)) # Find the nearest male for each female.
            dist_MF_list.extend(dists_fm.min(axis=0)) # Find the nearest female for each male.

    # Compute total travel distances.
    travel_dist = {'Female': 0.0, 'Male': 0.0}
    for id_p, p_data in df_iv.groupby('id_p'):
        p_data = p_data.sort_values('frame')
        coords = p_data[['x', 'y']].values
        if len(coords) > 1:
            # Compute displacement between consecutive frames.
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

# Break threshold: if a group disappears for more than 2 seconds (60 frames at 30 FPS),
# treat it as dissolved. Its next appearance starts a new "Session".
MAX_GAP_FRAMES = int(FPS * 2.0) 

for id_g, g_data in df_valid_groups.groupby('id_g'):
    # Get and sort the frames in which this group ID appears.
    unique_frames = sorted(g_data['frame'].unique())
    
    # Split frames into continuous sessions.
    sessions = []
    current_session = [unique_frames[0]]
    
    for i in range(1, len(unique_frames)):
        # If the gap between two frames exceeds the threshold, break the session.
        if unique_frames[i] - unique_frames[i-1] > MAX_GAP_FRAMES:
            sessions.append(current_session)
            current_session = [unique_frames[i]]
        else:
            current_session.append(unique_frames[i])
    sessions.append(current_session) # Append the final session.
    
    # Compute metrics for each actual session.
    for session_idx, session_frames in enumerate(sessions):
        first_f = session_frames[0]
        last_f = session_frames[-1]
        
        duration_sec = (last_f - first_f) / FPS
        
        # If a session appears in only one frame (f_min equals f_max),
        # add the duration of FRAME_STRIDE (default: 3 frames ~ 0.1 s).
        if duration_sec == 0:
            duration_sec = 3 / FPS 
            
        # Filter noise: exclude groups lasting less than 1 second.
        if duration_sec < 1.0:
            continue
            
        # Select data belonging to the current session only.
        session_data = g_data[g_data['frame'].isin(session_frames)]
        
        unique_members = session_data.drop_duplicates(subset=['id_p'])
        f_count = sum(unique_members['gender'] == 'Female')
        m_count = sum(unique_members['gender'] == 'Male')
        size = f_count + m_count
        
        group_stats.append({
            'Group ID': f"{id_g} (Session {session_idx + 1})", # Distinguish IDs by gathering session.
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
    
    # Metadata sheet to explain results to collaborators.
    meta_df = pd.DataFrame([
        {'Variable': 'Group ID', 'Explanation': 'Unique identifier for a social cluster. ID > 0 implies interaction. ID = -1 means the person is isolated.'},
        {'Variable': 'Space Occupied (m2)', 'Explanation': 'Calculated using Convex Hull algorithm over the real-world coordinates.'},
        {'Variable': 'Var Dist', 'Explanation': 'Variance of the distance, representing the extent of variation over the period.'}
    ])
    meta_df.to_excel(writer, sheet_name='Metadata & Clarifications', index=False)

print("✅ Hoàn thành xuất Report Analytics.")