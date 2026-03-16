import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CẤU HÌNH ---
INPUT_PKL = "./output_offline_3d_poses/PETS09-S2L1_3d_poses_MixSTE.pkl"
OUTPUT_IMG_DIR = "./output_visualizations"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Khớp xương chuẩn COCO 17 (Nối các điểm lại thành khung xương người)
# Khớp xương chuẩn Human3.6M (Do MixSTE nhả ra)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),          # Chân phải (Hông -> Đầu gối -> Cổ chân)
    (0, 4), (4, 5), (5, 6),          # Chân trái (Hông -> Đầu gối -> Cổ chân)
    (0, 7), (7, 8), (8, 9), (9, 10), # Trục dọc (Hông -> Cột sống -> Cổ -> Đầu)
    (8, 11), (11, 12), (12, 13),     # Tay trái (Cổ -> Vai -> Khuỷu tay -> Cổ tay)
    (8, 14), (14, 15), (15, 16)      # Tay phải (Cổ -> Vai -> Khuỷu tay -> Cổ tay)
]

def plot_3d_skeleton(pose_3d, track_id, frame_id, save_path):
    # Khởi tạo không gian 3D
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Trong Computer Vision, trục Y thường hướng xuống đất.
    # Nên để vẽ lên đồ thị 3D cho người đứng thẳng, ta đảo chiều trục Y.
    # Ánh xạ: X_plot = X, Y_plot = Z (chiều sâu), Z_plot = -Y (chiều cao)
    x = pose_3d[:, 0]
    y = pose_3d[:, 2] 
    z = -pose_3d[:, 1] 

    # 1. Vẽ các "khớp" (Joints)
    ax.scatter(x, y, z, c='red', s=40, edgecolors='black', label='Joints')

    # 2. Vẽ các "xương" (Bones)
    for (i, j) in CONNECTIONS:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], c='blue', linewidth=3)

    # 3. Làm đẹp đồ thị
    ax.set_title(f"3D Skeleton - Track ID: {track_id} | Frame: {frame_id}", fontsize=15, fontweight='bold')
    ax.set_xlabel('X-axis (Width)')
    ax.set_ylabel('Z-axis (Depth)')
    ax.set_zlabel('Y-axis (Height)')
    
    # Ép tỷ lệ 3 trục bằng nhau để người không bị lùn/méo
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Chỉnh góc nhìn hơi chéo từ trên xuống để thấy rõ 3D (Elevation=20, Azimuth=45)
    ax.view_init(elev=20, azim=45)

    # Lưu ảnh
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu ảnh 3D tại: {save_path}")

# --- ĐỌC VÀ VẼ ---
print(f"📦 Đang đọc dữ liệu 3D từ: {INPUT_PKL}")
with open(INPUT_PKL, 'rb') as f:
    data_3d = pickle.load(f)

# Chọn một người có nhiều frame (Ví dụ ID 4 có 363 frames)
target_id = 4
if target_id in data_3d:
    frames = sorted(list(data_3d[target_id].keys()))
    
    # Lấy 1 frame ở giữa lúc người này đang lọt rõ vào camera
    test_frame = frames[len(frames) // 2]
    pose_3d = data_3d[target_id][test_frame]
    
    print(f"🎨 Đang vẽ khung xương cho ID {target_id} tại frame {test_frame}...")
    save_img_path = os.path.join(OUTPUT_IMG_DIR, f"3d_skeleton_id{target_id}_frame{test_frame}.png")
    plot_3d_skeleton(pose_3d, target_id, test_frame, save_img_path)
else:
    print(f"❌ Không tìm thấy ID {target_id} trong dữ liệu.")