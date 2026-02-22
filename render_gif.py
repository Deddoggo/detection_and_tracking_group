import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

# --- CẤU HÌNH ---
INPUT_PKL = "./output_offline_3d_poses/PETS09-S2L1_3d_poses_MixSTE.pkl"
OUTPUT_IMG_DIR = "./output_visualizations"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Khớp xương chuẩn Human3.6M (MixSTE)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),          # Chân phải
    (0, 4), (4, 5), (5, 6),          # Chân trái
    (0, 7), (7, 8), (8, 9), (9, 10), # Trục dọc (Cột sống -> Đầu)
    (8, 11), (11, 12), (12, 13),     # Tay trái
    (8, 14), (14, 15), (15, 16)      # Tay phải
]

# --- ĐỌC DỮ LIỆU ---
print(f"📦 Đang đọc dữ liệu 3D từ: {INPUT_PKL}")
with open(INPUT_PKL, 'rb') as f:
    data_3d = pickle.load(f)

target_id = 4  # ID người bạn muốn làm video (ID 4 có 363 frames)
frames = sorted(list(data_3d[target_id].keys()))
poses_3d = [data_3d[target_id][f] for f in frames]

# --- SETUP KHÔNG GIAN 3D ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

def update(num):
    ax.clear()
    pose_3d = poses_3d[num]
    frame_id = frames[num]
    
    # [KỸ XẢO CAMERA TRACKING]: Dời tâm gốc tọa độ về điểm số 0 (Hông/Pelvis)
    # Giúp người luôn ở giữa màn hình dù họ đang đi dạo quanh camera
    pelvis = pose_3d[0]
    pose_centered = pose_3d - pelvis
    
    # Ánh xạ trục cho hiển thị Matplotlib (Y hướng lên)
    x = pose_centered[:, 0]  # Dim 0: Trục X (Ngang)
    y = pose_centered[:, 1]  # Dim 1: Trục Y (Chiều sâu)
    z = pose_centered[:, 2]  # Dim 2: Trục Z (Chiều cao - Tự hướng lên)
    
    # Vẽ khớp và xương
    ax.scatter(x, y, z, c='red', s=40, edgecolors='black')
    for (i, j) in CONNECTIONS:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c='blue', linewidth=3)
        
    ax.set_title(f"3D Skeleton Demo - MixSTE\nTrack ID: {target_id} | Frame: {frame_id}", fontweight='bold')
    ax.set_xlabel('Trục X (Ngang)')
    ax.set_ylabel('Trục Y (Chiều sâu)')
    ax.set_zlabel('Trục Z (Chiều cao)')
    
    # Khóa cứng tỷ lệ trục (+/- 1 mét) để không bị giật hình
    radius = 1.0 
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])
    
    # Xoay camera một góc nghiêng để nhìn 3D rõ nhất
    ax.view_init(elev=20, azim=45)

# --- RENDER GIF ---
print(f"🎬 Đang render GIF cho ID {target_id} ({len(frames)} frames)... Vui lòng đợi vài chục giây.")
# Interval=50 tương đương 20 fps (1000ms / 50 = 20)
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50)

gif_path = os.path.join(OUTPUT_IMG_DIR, f"demo_mixste_id{target_id}.gif")
ani.save(gif_path, writer='pillow') # Dùng pillow để xuất GIF

print(f"✅ Tuyệt vời! Đã lưu video GIF siêu mượt tại: {gif_path}")