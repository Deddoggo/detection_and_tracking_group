import pickle
import cv2
import numpy as np
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
INPUT_VIDEO = "./PETS09-S2L1-raw.webm" 
INPUT_2D_PKL = "./output_offline_2d_poses/PETS09-S2L1_2d_poses.pkl"
INPUT_3D_PKL = "./output_offline_3d_poses/PETS09-S2L1_3d_poses_MixSTE.pkl"

OUTPUT_DIR = "./output_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "PETS09_3D_Orientation.mp4")

# --- ĐỌC DỮ LIỆU ---
print("📦 Đang đọc dữ liệu 2D và 3D...")
with open(INPUT_2D_PKL, 'rb') as f:
    data_2d = pickle.load(f)
with open(INPUT_3D_PKL, 'rb') as f:
    data_3d = pickle.load(f)

# --- KHỞI TẠO VIDEO XUẤT ---
cap = cv2.VideoCapture(INPUT_VIDEO)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# --- HÀM TÍNH TOÁN VECTOR HƯỚNG ---
def get_chest_normal_vector(pose_3d):
    """
    Tính Vector pháp tuyến đâm ra từ lồng ngực dựa trên Tích có hướng.
    pose_3d: np.array (17, 3) theo chuẩn Human3.6M
    """
    # 1. Trục dọc (Cột sống): Từ Hông (0) lên Cổ (8)
    v_spine = pose_3d[8] - pose_3d[0]
    
    # 2. Trục ngang (Bờ vai): Từ Vai Trái (11) sang Vai Phải (14)
    v_shoulder = pose_3d[11] - pose_3d[14]
    
    # 3. Tích có hướng: v_spine x v_shoulder = Mũi tên đâm ra lồng ngực
    v_normal = np.cross(v_spine, v_shoulder)
    
    # 4. Chuẩn hóa vector (đưa độ dài về 1)
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
        
    # Duyệt qua từng ID trong khung hình này
    for track_id in data_3d.keys():
        # Kiểm tra xem người này có dữ liệu ở frame hiện tại không
        if frame_id in data_3d[track_id] and frame_id in data_2d[track_id]:
            pose_3d = data_3d[track_id][frame_id]
            pose_2d = data_2d[track_id][frame_id] # Dữ liệu 2D là chuẩn COCO
            
            # --- TÍNH VECTOR 3D ---
            v_normal = get_chest_normal_vector(pose_3d)
            
            # Trên không gian 3D của MixSTE (World Coordinates):
            # Dim 0: Trục X (Chiều ngang)
            # Dim 1: Trục Y (Chiều sâu)
            # Vì camera PETS09 chúi xuống, hướng chiều sâu (Y 3D) tương đương với hướng đi lên/xuống (Y 2D)
            dx_3d = v_normal[0]
            dy_3d = v_normal[1] 
            
            # --- TÌM ĐIỂM GẮN MŨI TÊN (TÂM LỒNG NGỰC 2D) ---
            # Trong COCO: Vai trái (5), Vai phải (6). Lấy trung điểm làm gốc mũi tên.
            chest_x = int((pose_2d[5][0] + pose_2d[6][0]) / 2)
            chest_y = int((pose_2d[5][1] + pose_2d[6][1]) / 2)
            
            # --- ÁNH XẠ XUỐNG 2D ---
            ARROW_LENGTH = 60 # Độ dài mũi tên trên màn hình (pixel)
            
            # LƯU Ý: Trục Y của ảnh OpenCV đi xuống, còn trục Y (chiều sâu) của 3D đâm ra xa.
            # Nên mũi tên đâm ra xa (dy_3d > 0) thì phải đi LÊN trên màn hình (trừ đi dy).
            end_x = int(chest_x + dx_3d * ARROW_LENGTH)
            end_y = int(chest_y - dy_3d * ARROW_LENGTH) 
            
            # --- VẼ LÊN FRAME ---
            # Vẽ đường thẳng mũi tên màu Xanh Lá Cây sặc sỡ, độ dày 3
            cv2.arrowedLine(frame, (chest_x, chest_y), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
            
            # Gắn ID lên đầu để dễ theo dõi
            cv2.putText(frame, f"ID:{track_id}", (chest_x - 15, chest_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    out.write(frame)
    print(f"  > Đã vẽ frame {frame_id}", end='\r')
    frame_id += 1

cap.release()
out.release()
print(f"\n✅ Hoàn thành! Mời bạn xem video kết quả tại: {OUTPUT_VIDEO}")