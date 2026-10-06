# Allow both python -m PACKAGE.NAME and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import repo_path, checkpoint_path, external_path, output_path, load_paths
import sys as _dependency_sys
_dependency_sys.path.insert(0, external_path())
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_PKL = repo_path('./output_offline_3d_poses/PETS09-S2L1_3d_poses_MixSTE.pkl')
OUTPUT_IMG_DIR = repo_path('./output_visualizations')
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# COCO 17 skeleton connections (connect joints into a human skeleton).
# Human3.6M skeleton connections (MixSTE output).
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),          # Right leg (hip -> knee -> ankle).
    (0, 4), (4, 5), (5, 6),          # Left leg (hip -> knee -> ankle).
    (0, 7), (7, 8), (8, 9), (9, 10), # Vertical axis (hip -> spine -> neck -> head).
    (8, 11), (11, 12), (12, 13),     # Left arm (neck -> shoulder -> elbow -> wrist).
    (8, 14), (14, 15), (15, 16)      # Right arm (neck -> shoulder -> elbow -> wrist).
]

def plot_3d_skeleton(pose_3d, track_id, frame_id, save_path):
    # Initialize 3D space.
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # In computer vision, the Y axis usually points downward.
    # Invert Y to display an upright person in the 3D plot.
    # Mapping: X_plot = X, Y_plot = Z (depth), Z_plot = -Y (height).
    x = pose_3d[:, 0]
    y = pose_3d[:, 2] 
    z = -pose_3d[:, 1] 

    # 1. Draw joints.
    ax.scatter(x, y, z, c='red', s=40, edgecolors='black', label='Joints')

    # 2. Draw bones.
    for (i, j) in CONNECTIONS:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c='blue', linewidth=3)

    # 3. Adjust plot appearance.
    ax.set_title(f"3D Skeleton - Track ID: {track_id} | Frame: {frame_id}", fontsize=15, fontweight='bold')
    ax.set_xlabel('X-axis (Width)')
    ax.set_ylabel('Z-axis (Depth)')
    ax.set_zlabel('Y-axis (Height)')
    
    # Use equal scaling on all three axes to prevent a shortened/distorted person.
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Use an oblique view from above for clearer 3D (Elevation=20, Azimuth=45).
    ax.view_init(elev=20, azim=45)

    # Save the image.
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu ảnh 3D tại: {save_path}")

# --- LOAD AND DRAW ---
print(f"📦 Đang đọc dữ liệu 3D từ: {INPUT_PKL}")
with open(INPUT_PKL, 'rb') as f:
    data_3d = pickle.load(f)

# Choose a person with many frames (e.g., ID 4 has 363 frames).
target_id = 4
if target_id in data_3d:
    frames = sorted(list(data_3d[target_id].keys()))
    
    # Select a middle frame where this person is clearly visible.
    test_frame = frames[len(frames) // 2]
    pose_3d = data_3d[target_id][test_frame]
    
    print(f"🎨 Đang vẽ khung xương cho ID {target_id} tại frame {test_frame}...")
    save_img_path = os.path.join(OUTPUT_IMG_DIR, f"3d_skeleton_id{target_id}_frame{test_frame}.png")
    plot_3d_skeleton(pose_3d, target_id, test_frame, save_img_path)
else:
    print(f"❌ Không tìm thấy ID {target_id} trong dữ liệu.")