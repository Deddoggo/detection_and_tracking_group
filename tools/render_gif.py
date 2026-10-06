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
import matplotlib.animation as animation
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_PKL = repo_path('./output_offline_3d_poses/PETS09-S2L1_3d_poses_MixSTE.pkl')
OUTPUT_IMG_DIR = repo_path('./output_visualizations')
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Human3.6M skeleton connections (MixSTE).
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),          # Right leg.
    (0, 4), (4, 5), (5, 6),          # Left leg.
    (0, 7), (7, 8), (8, 9), (9, 10), # Vertical axis (spine -> head).
    (8, 11), (11, 12), (12, 13),     # Left arm.
    (8, 14), (14, 15), (15, 16)      # Right arm.
]

# --- LOAD DATA ---
print(f"📦 Đang đọc dữ liệu 3D từ: {INPUT_PKL}")
with open(INPUT_PKL, 'rb') as f:
    data_3d = pickle.load(f)

target_id = 4  # Person ID to visualize (ID 4 has 363 frames).
frames = sorted(list(data_3d[target_id].keys()))
poses_3d = [data_3d[target_id][f] for f in frames]

# --- SET UP 3D SPACE ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

def update(num):
    ax.clear()
    pose_3d = poses_3d[num]
    frame_id = frames[num]
    
    # [CAMERA TRACKING EFFECT]: move the coordinate origin to joint 0 (hip/pelvis).
    # Keep the person centered on screen while they move around the camera.
    pelvis = pose_3d[0]
    pose_centered = pose_3d - pelvis
    
    # Map axes for Matplotlib display (Y points upward).
    x = pose_centered[:, 0]  # Dim 0: X axis (horizontal).
    y = pose_centered[:, 1]  # Dim 1: Y axis (depth).
    z = pose_centered[:, 2]  # Dim 2: Z axis (height - already points upward).
    
    # Draw joints and bones.
    ax.scatter(x, y, z, c='red', s=40, edgecolors='black')
    for (i, j) in CONNECTIONS:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c='blue', linewidth=3)
        
    ax.set_title(f"3D Skeleton Demo - MixSTE\nTrack ID: {target_id} | Frame: {frame_id}", fontweight='bold')
    ax.set_xlabel('X-axis (Width)')
    ax.set_ylabel('Y-axis (Depth)')
    ax.set_zlabel('Z-axis (Height)')
    
    # Fix axis limits at +/- 1 meter to avoid jitter.
    radius = 1.0 
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])
    
    # Use an oblique camera angle for a clearer 3D view.
    ax.view_init(elev=20, azim=45)

# --- RENDER GIF ---
print(f"🎬 Đang render GIF cho ID {target_id} ({len(frames)} frames)... Vui lòng đợi vài chục giây.")
# Interval=50 corresponds to 20 fps (1000 ms / 50 = 20).
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50)

gif_path = os.path.join(OUTPUT_IMG_DIR, f"demo_mixste_id{target_id}.gif")
ani.save(gif_path, writer='pillow') # Export the GIF using Pillow.

print(f"✅ Tuyệt vời! Đã lưu video GIF siêu mượt tại: {gif_path}")