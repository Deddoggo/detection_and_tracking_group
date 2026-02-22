import os
import pickle
import numpy as np
import torch
import sys
import argparse

# Thêm đường dẫn tuyệt đối để import mô hình MixSTE
sys.path.append(os.path.abspath('./MixSTE'))
from common.model_cross import MixSTE2

# --- CẤU HÌNH ---
INPUT_PKL = "./output_offline_2d_poses/PETS09-S2L1_2d_poses.pkl"
OUTPUT_3D_DIR = "./output_offline_3d_poses"
os.makedirs(OUTPUT_3D_DIR, exist_ok=True)

# Khung thời gian tiêu chuẩn của MixSTE để đạt SOTA (thường là 243)
FRAMES_WINDOW = 243  
PAD_HALF = FRAMES_WINDOW // 2

def convert_coco_to_h36m(coco_points):
    """
    Chuyển đổi 17 điểm chuẩn COCO (ViTPose) sang 17 điểm chuẩn Human3.6M (MixSTE)
    """
    h36m = np.zeros((17, 2), dtype=np.float32)
    
    # 0: Pelvis (Tâm của 2 hông COCO 11 và 12)
    h36m[0] = (coco_points[11] + coco_points[12]) / 2.0
    
    # Chân phải: 1: R_Hip(12), 2: R_Knee(14), 3: R_Ankle(16)
    h36m[1], h36m[2], h36m[3] = coco_points[12], coco_points[14], coco_points[16]
    
    # Chân trái: 4: L_Hip(11), 5: L_Knee(13), 6: L_Ankle(15)
    h36m[4], h36m[5], h36m[6] = coco_points[11], coco_points[13], coco_points[15]
    
    # 8: Neck/Thorax (Tâm của 2 vai COCO 5 và 6)
    h36m[8] = (coco_points[5] + coco_points[6]) / 2.0
    
    # 7: Spine (Tâm của Pelvis và Neck)
    h36m[7] = (h36m[0] + h36m[8]) / 2.0
    
    # 9: Đầu/Mũi (COCO 0)
    h36m[9] = coco_points[0]
    
    # 10: Đỉnh đầu (Ước lượng vươn lên từ cổ qua mũi)
    h36m[10] = coco_points[0] + (coco_points[0] - h36m[8]) * 0.5 
    
    # Tay trái: 11: L_Shoulder(5), 12: L_Elbow(7), 13: L_Wrist(9)
    h36m[11], h36m[12], h36m[13] = coco_points[5], coco_points[7], coco_points[9]
    
    # Tay phải: 14: R_Shoulder(6), 15: R_Elbow(8), 16: R_Wrist(10)
    h36m[14], h36m[15], h36m[16] = coco_points[6], coco_points[8], coco_points[10]
    
    return h36m
    
# --- 1. HÀM CHUẨN BỊ CHUỖI THỜI GIAN (TEMPORAL PADDING) ---
def prepare_sequence(pose_dict, track_id):
    frames = sorted(list(pose_dict.keys()))
    if len(frames) == 0: return None, None
    
    start_f, end_f = frames[0], frames[-1]
    total_frames = end_f - start_f + 1
    
    seq_2d = np.zeros((total_frames, 17, 2), dtype=np.float32)
    valid_mask = np.zeros(total_frames, dtype=bool)
    
    for f_id in frames:
        idx = f_id - start_f
        seq_2d[idx] = convert_coco_to_h36m(pose_dict[f_id][:, :2])
        valid_mask[idx] = True
        
    # Nội suy tuyến tính cho các frame bị mất dấu (missed detection)
    for i in range(17): 
        for j in range(2): 
            seq_2d[:, i, j] = np.interp(np.arange(total_frames), 
                                        np.where(valid_mask)[0], 
                                        seq_2d[valid_mask, i, j])
            
    return seq_2d, frames

# --- 2. KHỞI TẠO MÔ HÌNH MIXSTE ---
print("🚀 Đang khởi tạo kiến trúc MixSTE (243 frames)...")
model = MixSTE2(
    num_frame=FRAMES_WINDOW, 
    num_joints=17, 
    in_chans=2, 
    embed_dim_ratio=512,
    depth=8,
    num_heads=8, 
    mlp_ratio=2., 
    qkv_bias=True, 
    qk_scale=None,
    drop_path_rate=0
)

if torch.cuda.is_available():
    model = model.cuda()

# [QUAN TRỌNG]: Nạp "tấm bằng đại học" vào mô hình
print("🧠 Đang nạp Pre-trained Weights (best_epoch_cpn_243f.bin)...")
checkpoint = torch.load('best_epoch_cpn_243f.bin', map_location='cuda:0', weights_only=False) 

# Lấy dict weights ra
state_dict = checkpoint['model_pos'] if 'model_pos' in checkpoint else checkpoint

# --- ĐOẠN CODE CẠO CHỮ 'module.' ---
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k # Bỏ 7 ký tự đầu ('module.')
    new_state_dict[name] = v
# -----------------------------------

# Nạp weights đã gọt sạch vào mô hình
model.load_state_dict(new_state_dict) 
model.eval()

def normalize_screen_coordinates(X, w, h):
    """Chuẩn hóa tọa độ pixel về khoảng [-1, 1] cho mạng nơ-ron"""
    assert X.shape[-1] == 2
    return X / w * 2 - np.array([1, h / w], dtype=X.dtype)

# --- 3. ĐỌC VÀ XỬ LÝ DỮ LIỆU OFFLINE ---
print(f"📦 Đang đọc dữ liệu 2D từ: {INPUT_PKL}")
with open(INPUT_PKL, 'rb') as f:
    offline_data = pickle.load(f)

output_3d_data = {}
print(f"Bắt đầu nâng 2D lên 3D bằng MixSTE cho {len(offline_data)} đối tượng...")

with torch.no_grad():
    for track_id, pose_dict in offline_data.items():
        seq_2d, original_frames = prepare_sequence(pose_dict, track_id)
        if seq_2d is None: continue
            
        # 1. Điền CHÍNH XÁC độ phân giải video
        VIDEO_W, VIDEO_H = 768, 576 
        
        # 2. Chuẩn hóa về khoảng [-1, 1] để mạng dễ đọc
        seq_2d_norm = normalize_screen_coordinates(seq_2d, VIDEO_W, VIDEO_H)
        
        # 3. [CHÌA KHÓA QUAN TRỌNG NHẤT]: Trừ đi Hông để dời gốc tọa độ về (0,0)
        pelvis_2d = seq_2d_norm[:, 0:1, :] 
        seq_2d_centered = seq_2d_norm - pelvis_2d
        
        T = seq_2d_centered.shape[0]
        output_3d_data[track_id] = {}
        
        # 4. Padding 2 đầu cửa sổ
        padded_seq = np.pad(seq_2d_centered, ((PAD_HALF, PAD_HALF), (0, 0), (0, 0)), mode='edge')
        
        # Batching: Ném từng cửa sổ vào MixSTE
        for i in range(T):
            window = padded_seq[i : i + FRAMES_WINDOW]
            
            input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            if torch.cuda.is_available(): input_tensor = input_tensor.cuda()
            
            pred_3d = model(input_tensor) 
            
            center_3d_pose = pred_3d.cpu().numpy()[0, PAD_HALF]
            
            frame_id = original_frames[0] + i if i < len(original_frames) else original_frames[-1]
            output_3d_data[track_id][frame_id] = center_3d_pose
            
        print(f"   + Đã xử lý xong ID {track_id} ({T} frames)")

# --- 4. XUẤT KẾT QUẢ ---
out_file = os.path.join(OUTPUT_3D_DIR, "PETS09-S2L1_3d_poses_MixSTE.pkl")
with open(out_file, 'wb') as f:
    pickle.dump(output_3d_data, f)

print(f"\n✅ Hoàn thành! Dữ liệu 3D siêu chuẩn đã lưu tại: {out_file}")