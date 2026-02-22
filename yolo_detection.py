import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import torch
from pathlib import Path
import time
import json
import math

# --- IMPORT BOXMOT ---
# DeepOCSORT: Thuật toán tốt nhất hiện nay cho Occlusion (vật thể bị che khuất)
from boxmot import DeepOCSORT

# Import hàm grouping của bạn
from identify_group import cluster_bboxes_with_ids

# --- CẤU HÌNH ---
REID_MODEL = 'osnet_x1_0_msmt17.pt' # BoxMOT sẽ tự tải file này nếu chưa có

# --- PARSE ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--input_video', type=str, required=True, help='Path to input video file')
parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder')
parser.add_argument('--calibration_file', type=str, default='calibration_matrix.json', help='Path to calibration matrix JSON')
parser.add_argument('--scale_factor', type=float, default=0.35, help='Correction factor for distances')
parser.add_argument('--epsilon', type=float, default=1.5, help='Distance threshold (Meters)')
parser.add_argument('--threshold_overlap', type=float, default=0.7, help='Overlap threshold')
args = parser.parse_args()

# Gán biến
input_video = args.input_video
output_folder = args.output_folder
calibration_file = args.calibration_file
scale_factor_val = args.scale_factor
epsilon = args.epsilon
threshold_overlap = args.threshold_overlap

os.makedirs(output_folder, exist_ok=True)

# --- 1. SETUP THIẾT BỊ (QUAN TRỌNG) ---
# YOLO cần string 'cuda:0', BoxMOT cần int 0. Phải tách ra để không lỗi.
if torch.cuda.is_available():
    device_yolo = 'cuda:0'
    device_boxmot = 0
    print("🟢 Running on GPU")
else:
    device_yolo = 'cpu'
    device_boxmot = 'cpu'
    print("🟡 Running on CPU")

# --- 2. LOAD CALIBRATION MATRIX ---
homography_matrix = None
if os.path.exists(calibration_file):
    try:
        with open(calibration_file, 'r') as f:
            data = json.load(f)
            homography_matrix = np.array(data['homography_matrix'], dtype=np.float32)
        print(f"✅ Matrix loaded. Scale Factor: {scale_factor_val}")
    except Exception as e:
        print(f"⚠️ Error loading matrix: {e}")
else:
    print(f"⚠️ Matrix file not found.")

# --- 3. LOAD MODELS ---
print("Loading YOLO model...")
try:
    # Chỉ dùng YOLO để Detect (phát hiện), không dùng để Track
    model = YOLO('yolo12x.pt') 
except:
    print("yolo12x.pt not found, using yolov8m.pt...")
    model = YOLO('yolov8m.pt')
model.to(device_yolo)

print(f"Initializing DeepOCSORT Tracker with {REID_MODEL}...")
# Cấu hình Tracker DeepOCSORT
tracker = DeepOCSORT(
    model_weights=Path(REID_MODEL), 
    device=device_boxmot,
    fp16=False,      # Tăng tốc độ
    det_thresh=0.4, # Chỉ track những vật thể rõ ràng (conf > 0.4)
    max_age=100,    # Giữ ID trong 100 frames (5 giây) nếu bị che khuất
    w_association_emb=0.75,
)

# --- 4. VIDEO SETUP ---
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"Error: Cannot open video {input_video}")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = os.path.join(output_folder, 'output_boxmot_deepocsort.mp4')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

groups_status = {}
max_group_id = -1
start_time = time.time()
frame_idx = 0

print(f"Start processing video: {input_video}...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # --- A. DETECT (YOLO PREDICT) ---
    # Thay đổi quan trọng: Dùng predict() chứ không phải track()
    results = model.predict(frame, conf=0.4, verbose=False, classes=0)[0]
    
    # Lấy dữ liệu boxes: [x1, y1, x2, y2, conf, class_id]
    dets = results.boxes.data.cpu().numpy()

    # --- B. TRACK (BOXMOT UPDATE) ---
    if len(dets) == 0:
        tracks = tracker.update(np.empty((0, 6)), frame)
    else:
        tracks = tracker.update(dets, frame)

    # --- C. PREPARE DATA FOR GROUPING ---
    boxes_xywh = []
    ids = []
    
    # BoxMOT trả về: [x1, y1, x2, y2, id, conf, cls, ind]
    if len(tracks) > 0:
        # Lọc bỏ những track chưa có ID (id == -1)
        valid_tracks = tracks[tracks[:, 4] != -1]
        
        for trk in valid_tracks:
            x1, y1, x2, y2 = trk[:4]
            track_id = int(trk[4])
            
            # Convert [x1, y1, x2, y2] -> [x, y, w, h] cho hàm grouping
            w = x2 - x1
            h = y2 - y1
            
            boxes_xywh.append([x1, y1, w, h])
            ids.append(track_id)

    # --- D. GROUPING LOGIC ---
    cluster_results, groups_status, max_group_id = cluster_bboxes_with_ids(
        groups_status, 
        boxes_xywh, 
        ids, 
        max_group_id, 
        homography_matrix=homography_matrix, 
        scale_factor=scale_factor_val,
        eps=epsilon, 
        threshold_overlap=threshold_overlap
    )

    # --- E. DRAWING ---
    # Vẽ bbox từng người
    for person in cluster_results:
        bbox = person['bbox']
        color = person['color']
        id_p = person['id_p']
        id_g = person['id_g']
        
        # Vẽ Box
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), color, 2)
        
        # Vẽ ID (Có nền đen)
        text = f"ID:{id_p} G:{id_g}" if id_g != -1 else f"ID:{id_p}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])-th-5), (int(bbox[0])+tw, int(bbox[1])), color, -1)
        cv2.putText(frame, text, (int(bbox[0]), int(bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Vẽ khoảng cách
    if homography_matrix is not None:
        num_people = len(cluster_results)
        for i in range(num_people):
            for j in range(i + 1, num_people):
                p1 = cluster_results[i]
                p2 = cluster_results[j]
                
                pos1 = p1.get('real_pos')
                pos2 = p2.get('real_pos')
                
                if pos1 is not None and pos2 is not None:
                    # Tính khoảng cách Euclidean
                    dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    if dist < 3.0:
                        # Lấy tâm bbox
                        c1_x = int(p1['bbox'][0] + p1['bbox'][2]/2)
                        c1_y = int(p1['bbox'][1] + p1['bbox'][3]/2)
                        c2_x = int(p2['bbox'][0] + p2['bbox'][2]/2)
                        c2_y = int(p2['bbox'][1] + p2['bbox'][3]/2)
                        
                        # Vẽ đường nối
                        cv2.line(frame, (c1_x, c1_y), (c2_x, c2_y), (203, 192, 255), 2)
                        
                        # Vẽ số mét
                        mid_x = (c1_x + c2_x) // 2
                        mid_y = (c1_y + c2_y) // 2
                        dist_text = f"{dist:.2f}m"
                        
                        (tw, th), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (mid_x, mid_y - th), (mid_x + tw, mid_y + 5), (0,0,0), -1)
                        cv2.putText(frame, dist_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Hiển thị thông tin Debug
    cv2.putText(frame, f"Mode: DeepOCSORT + OSNet", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    print(f"Processed frame {frame_idx} | Tracks: {len(ids)}", end='\r')
    out.write(frame)
    frame_idx += 1

end_time = time.time()
total_time = end_time - start_time
print(f"\n\nProcessing done! Total time: {total_time:.2f}s")
print(f"Output saved to: {output_path}")

out.release()
cap.release()