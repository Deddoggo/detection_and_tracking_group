# Allow both python -m PACKAGE.NAME and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import repo_path, checkpoint_path, external_path, output_path, load_paths
import sys as _dependency_sys
_dependency_sys.path.insert(0, external_path())
import pickle
import cv2
import numpy as np
import os

# --- PATH CONFIGURATION ---
INPUT_VIDEO = "./PETS09-S2L1-raw.webm" 
INPUT_2D_PKL = repo_path('./output_offline_2d_poses/PETS09-S2L1_2d_poses.pkl')
INPUT_3D_PKL = repo_path('./output_offline_3d_poses/PETS09-S2L1_3d_poses_MixSTE.pkl')

OUTPUT_DIR = repo_path('./output_visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "PETS09_3D_Orientation.mp4")

# --- LOAD DATA ---
print("📦 Đang đọc dữ liệu 2D và 3D...")
with open(INPUT_2D_PKL, 'rb') as f:
    data_2d = pickle.load(f)
with open(INPUT_3D_PKL, 'rb') as f:
    data_3d = pickle.load(f)

# --- INITIALIZE OUTPUT VIDEO ---
cap = cv2.VideoCapture(INPUT_VIDEO)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# --- DIRECTION VECTOR COMPUTATION ---
def get_chest_normal_vector(pose_3d):
    """Compute the outward chest normal using a cross product.
    pose_3d: np.array (17, 3) in the Human3.6M convention."""
    # 1. Vertical axis (spine): from hip (0) to neck (8).
    v_spine = pose_3d[8] - pose_3d[0]
    
    # 2. Horizontal axis (shoulders): from left shoulder (11) to right shoulder (14).
    v_shoulder = pose_3d[11] - pose_3d[14]
    
    # 3. Cross product: v_spine x v_shoulder gives a vector pointing outward from the chest.
    v_normal = np.cross(v_spine, v_shoulder)
    
    # 4. Normalize the vector to unit length.
    norm = np.linalg.norm(v_normal)
    if norm > 1e-6:
        v_normal = v_normal / norm
    else:
        v_normal = np.array([0, 0, 0])
        
    return v_normal

print("🎬 Bắt đầu vẽ Vector hướng lên Video...")
frame_id = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Iterate over each ID in this frame.
    for track_id in data_3d.keys():
        # Check whether this person has data in the current frame.
        if frame_id in data_3d[track_id] and frame_id in data_2d[track_id]:
            pose_3d = data_3d[track_id][frame_id]
            pose_2d = data_2d[track_id][frame_id] # The 2D data follow the COCO convention.
            
            # --- COMPUTE 3D VECTOR ---
            v_normal = get_chest_normal_vector(pose_3d)
            
            # In MixSTE 3D space (world coordinates):
            # Dim 0: X axis (horizontal).
            # Dim 1: Y axis (depth).
            # With the downward-facing PETS09 camera, depth (3D Y) corresponds to up/down motion (2D Y).
            dx_3d = v_normal[0]
            dy_3d = v_normal[1] 
            
            # --- FIND THE ARROW ORIGIN (2D CHEST CENTER) ---
            # COCO: left shoulder (5), right shoulder (6). Use their midpoint as the arrow origin.
            chest_x = int((pose_2d[5][0] + pose_2d[6][0]) / 2)
            chest_y = int((pose_2d[5][1] + pose_2d[6][1]) / 2)
            
            # --- MAP TO 2D ---
            ARROW_LENGTH = 60 # Arrow length on screen (pixels).
            
            # NOTE: OpenCV image Y points downward, while 3D depth Y points away from the camera.
            # An arrow pointing away (dy_3d > 0) must point UP on screen (subtract dy).
            end_x = int(chest_x + dx_3d * ARROW_LENGTH)
            end_y = int(chest_y - dy_3d * ARROW_LENGTH) 
            
            # --- DRAW ON FRAME ---
            # Draw a bright green arrow with thickness 3.
            cv2.arrowedLine(frame, (chest_x, chest_y), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
            
            # Place the ID above the head for easier tracking.
            cv2.putText(frame, f"ID:{track_id}", (chest_x - 15, chest_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    out.write(frame)
    print(f"  > Đã vẽ frame {frame_id}", end='\r')
    frame_id += 1

cap.release()
out.release()
print(f"\n✅ Hoàn thành! Mời bạn xem video kết quả tại: {OUTPUT_VIDEO}")