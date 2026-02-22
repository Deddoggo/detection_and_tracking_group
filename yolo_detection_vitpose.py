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
from collections import deque

# --- IMPORT BOXMOT ---
from boxmot import create_tracker
from identify_group import cluster_bboxes_with_ids
from orientation_utils import calculate_iou, get_pose_vector, cosine_similarity

# --- IMPORT HUGGING FACE ---
try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation
except ImportError:
    print("⚠️ Lỗi: Chưa cài transformers. Chạy: pip install --upgrade transformers accelerate timm")

# --- CLASS WRAPPER CHO HUGGING FACE VITPOSE (ĐÃ CẬP NHẬT THEO TÀI LIỆU) ---
class HFViTPosePredictor:
    def __init__(self, model_name="usyd-community/vitpose-base-simple", device='cuda:0'):
        print(f"Initializing HF ViTPose ({model_name})...")
        try:
            self.device = device
            # trust_remote_code=True là bắt buộc cho model ViTPose community
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = VitPoseForPoseEstimation.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.model.eval()
            print("✅ HF ViTPose initialized successfully.")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo HF ViTPose: {e}")
            print("👉 Hãy đảm bảo bạn đã cài: pip install transformers accelerate timm")
            self.model = None

    def predict(self, img_rgb, bboxes_xyxy):
        """
        img_rgb: Ảnh gốc RGB
        bboxes_xyxy: List các bbox [x1, y1, x2, y2] từ YOLO
        """
        if self.model is None or len(bboxes_xyxy) == 0:
            return [None] * len(bboxes_xyxy)

        try:
            # 1. Chuyển đổi Box từ XYXY (YOLO) sang XYWH (COCO) cho ViTPose
            boxes_coco = []
            for box in bboxes_xyxy:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                boxes_coco.append([x1, y1, w, h])

            # 2. Preprocess (Tự động crop & resize)
            # boxes phải là list của list các box (vì processor hỗ trợ batch nhiều ảnh)
            inputs = self.processor(img_rgb, boxes=[boxes_coco], return_tensors="pt").to(self.device)

            # 3. Inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 4. Post-process (Giải mã Heatmap -> Keypoints)
            # Trả về list cho từng ảnh, ta lấy ảnh đầu tiên [0]
            pose_results = self.processor.post_process_pose_estimation(outputs, boxes=[boxes_coco])[0]
            
            # 5. Format lại output để khớp với logic vẽ hình
            final_keypoints = []
            for res in pose_results:
                # res là dict: {'keypoints': tensor[17, 2], 'scores': tensor[17]}
                kpts = res['keypoints'].cpu().numpy()
                scores = res['scores'].cpu().numpy()
                
                # Gom lại thành [17, 3] (x, y, conf)
                kpts_with_conf = np.zeros((len(kpts), 3))
                kpts_with_conf[:, :2] = kpts
                kpts_with_conf[:, 2] = scores
                
                final_keypoints.append(kpts_with_conf)
                
            return final_keypoints

        except Exception as e:
            print(f"⚠️ HF ViTPose Error: {e}")
            import traceback
            traceback.print_exc()
            return [None] * len(bboxes_xyxy)

# --- CẤU HÌNH CHUNG ---
REID_MODEL = 'osnet_x1_0_msmt17.pt' 

# --- PARSE ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--input_video', type=str, required=True)
parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument('--calibration_file', type=str, default='calibration_matrix.json')
parser.add_argument('--scale_factor', type=float, default=0.35)
parser.add_argument('--epsilon', type=float, default=1.5)
parser.add_argument('--min_samples', type=int, default=2) # Fix lỗi DBSCAN
parser.add_argument('--vitpose_model', type=str, default='usyd-community/vitpose-base-simple')

args = parser.parse_args()
os.makedirs(args.output_folder, exist_ok=True)

# 1. SETUP THIẾT BỊ
if torch.cuda.is_available():
    device_yolo = 'cuda:0'
    device_boxmot = 0
    print("🟢 Running on GPU")
else:
    device_yolo = 'cpu'
    device_boxmot = 'cpu'
    print("🟡 Running on CPU")

# 2. LOAD CALIBRATION
homography_matrix = None
if os.path.exists(args.calibration_file):
    try:
        with open(args.calibration_file, 'r') as f:
            data = json.load(f)
            homography_matrix = np.array(data['homography_matrix'], dtype=np.float32)
    except: pass

# 3. LOAD MODELS
print("Loading YOLO Detection model...")
try:
    model = YOLO('yolo12x.pt') # Dùng bản Detection (x) mạnh nhất
except:
    model = YOLO('yolo26x.pt')
model.to(device_yolo)

# Load HF ViTPose
vitpose = HFViTPosePredictor(model_name=args.vitpose_model, device=device_yolo)

print(f"Initializing DeepOCSORT Tracker...")
tracker = create_tracker(
    tracker_type='deepocsort',
    tracker_config=None,          # dùng default
    reid_weights='osnet_x1_0_msmt17.pt',
    device=device_boxmot,
    half=True
)

# 4. VIDEO SETUP
cap = cv2.VideoCapture(args.input_video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = os.path.join(args.output_folder, 'output_yolo_vitpose_final.mp4')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# VARIABLES
track_history = {} 
frame_idx = 0
groups_status = {}
max_group_id = -1

# VISUALIZATION CONFIG
VISUAL_FONT_SCALE = 0.45 
VISUAL_THICKNESS = 1     
ARROW_LENGTH = 30         
ARROW_THICKNESS = 2      
ARROW_TIP = 0.3         

# Thêm biến lưu trạng thái Pose để làm mượt (Smoothing)
pose_history_smooth = {} # {track_id: (vx, vy)}
ALPHA_SMOOTH = 0.7       # Hệ số làm mượt (Càng nhỏ càng mượt nhưng trễ)

print(f"Start processing...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # A. Detect
    results = model.predict(frame, conf=0.25, imgsz=1280, verbose=False, classes=0)[0]
    dets = results.boxes.data.cpu().numpy() 

    # B. Track
    if len(dets) == 0:
        tracks = tracker.update(np.empty((0, 6)), frame)
    else:
        tracks = tracker.update(dets, frame)

    # C. Prepare Data
    boxes_xywh = []
    ids = []
    valid_tracks_list = [] 
    bboxes_for_pose = []  

    current_frame_ids = set() # Để dọn dẹp bộ nhớ smoothing

    if len(tracks) > 0:
        valid_tracks = tracks[tracks[:, 4] != -1]
        for trk in valid_tracks:
            x1, y1, x2, y2 = trk[:4]
            tid = int(trk[4])
            
            bboxes_for_pose.append([x1, y1, x2, y2])
            valid_tracks_list.append(trk)
            current_frame_ids.add(tid)

    # Dọn dẹp id cũ khỏi bộ nhớ smoothing
    pose_history_smooth = {k: v for k, v in pose_history_smooth.items() if k in current_frame_ids}

    # D. Pose Estimation & Smoothing
    active_tracks_info = {}
    
    if len(bboxes_for_pose) > 0:
        pose_results = vitpose.predict(frame_rgb, bboxes_for_pose)
        
        for i, trk in enumerate(valid_tracks_list):
            tid = int(trk[4])
            x1, y1, x2, y2 = trk[:4]
            cx, cy = (x1 + x2)/2, (y1 + y2)/2
            
            kpts = pose_results[i] if i < len(pose_results) else None
            
            # --- LOGIC MỚI: SMOOTHING POSE VECTOR ---
            raw_pose_vec = get_pose_vector(kpts, conf_thresh=0.6) # Tăng thresh lên 0.6 cho chắc
            
            final_pose_vec = None
            
            if raw_pose_vec is not None:
                if tid in pose_history_smooth:
                    # Công thức EMA: New = alpha * Raw + (1-alpha) * Old
                    old_vx, old_vy = pose_history_smooth[tid]
                    new_vx = ALPHA_SMOOTH * raw_pose_vec[0] + (1 - ALPHA_SMOOTH) * old_vx
                    new_vy = ALPHA_SMOOTH * raw_pose_vec[1] + (1 - ALPHA_SMOOTH) * old_vy
                    
                    # Chuẩn hóa lại
                    norm = math.sqrt(new_vx**2 + new_vy**2)
                    if norm > 0:
                        final_pose_vec = (new_vx/norm, new_vy/norm)
                        pose_history_smooth[tid] = final_pose_vec
                else:
                    pose_history_smooth[tid] = raw_pose_vec
                    final_pose_vec = raw_pose_vec
            else:
                # Nếu frame này mất pose (do che khuất), dùng lại hướng cũ nếu có
                if tid in pose_history_smooth:
                    final_pose_vec = pose_history_smooth[tid]

            # --- MOTION VECTOR ---
            if tid not in track_history: track_history[tid] = deque(maxlen=30)
            track_history[tid].append((cx, cy))
            
            motion_vec = None
            hist = track_history[tid]
            if len(hist) >= 10: # Tăng lên 10 frame để vector dài và ổn định hơn
                vx = hist[-1][0] - hist[0][0]
                vy = hist[-1][1] - hist[0][1]
                norm_v = math.sqrt(vx**2 + vy**2)
                if norm_v > 5: # Chỉ vẽ nếu di chuyển > 5px
                    motion_vec = (vx/norm_v, vy/norm_v)

            active_tracks_info[tid] = {
                'pose_vec': final_pose_vec,
                'motion_vec': motion_vec,
                'center': (cx, cy)
            }
            
            w = x2 - x1
            h = y2 - y1
            boxes_xywh.append([x1, y1, w, h])
            ids.append(tid)

    # --- E. GROUPING LOGIC ---
    cluster_results, groups_status, max_group_id = cluster_bboxes_with_ids(
        groups_status, boxes_xywh, ids, max_group_id, 
        homography_matrix, args.scale_factor, args.epsilon, args.min_samples
    )

    # --- F. DRAWING ---
    for person in cluster_results:
        bbox = person['bbox']
        color = person['color']
        tid = person['id_p']
        id_g = person['id_g']
        
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), color, VISUAL_THICKNESS)
        
        text = f"{tid}" if id_g == -1 else f"{tid}-G{id_g}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, VISUAL_FONT_SCALE, VISUAL_THICKNESS)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])-th-4), (int(bbox[0])+tw+2, int(bbox[1])), color, -1)
        cv2.putText(frame, text, (int(bbox[0])+1, int(bbox[1])-2), cv2.FONT_HERSHEY_SIMPLEX, VISUAL_FONT_SCALE, (255,255,255), VISUAL_THICKNESS)

        if tid in active_tracks_info:
            info = active_tracks_info[tid]
            cx, cy = int(info['center'][0]), int(info['center'][1])
            
            if info['pose_vec'] is not None:
                px, py = info['pose_vec']
                end_x = int(cx + px * ARROW_LENGTH)
                end_y = int(cy + py * ARROW_LENGTH)
                cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 255, 255), ARROW_THICKNESS, tipLength=ARROW_TIP)
            
            if info['motion_vec'] is not None:
                mx, my = info['motion_vec']
                end_mx = int(cx + mx * ARROW_LENGTH)
                end_my = int(cy + my * 40)
                cv2.arrowedLine(frame, (cx, cy), (end_mx, end_my), (0, 0, 255), ARROW_THICKNESS, tipLength=ARROW_TIP)

    if homography_matrix is not None:
        num_people = len(cluster_results)
        for i in range(num_people):
            for j in range(i + 1, num_people):
                p1 = cluster_results[i]
                p2 = cluster_results[j]
                pos1, pos2 = p1.get('real_pos'), p2.get('real_pos')
                # Fix lỗi check array
                if pos1 is not None and pos2 is not None:
                    dist = math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
                    if dist < 3.0:
                        c1 = (int(p1['bbox'][0]+p1['bbox'][2]/2), int(p1['bbox'][1]+p1['bbox'][3]/2))
                        c2 = (int(p2['bbox'][0]+p2['bbox'][2]/2), int(p2['bbox'][1]+p2['bbox'][3]/2))
                        cv2.line(frame, c1, c2, (200, 200, 200), 1)
                        mid = ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2)
                        dist_text = f"{dist:.1f}m"
                        (tw, th), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                        cv2.rectangle(frame, (mid[0], mid[1]-th), (mid[0]+tw, mid[1]+2), (0,0,0), -1)
                        cv2.putText(frame, dist_text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    print(f"Processed frame {frame_idx} | Tracks: {len(ids)}", end='\r')
    out.write(frame)
    frame_idx += 1

out.release()
cap.release()
print(f"\n\nDone! Saved to: {output_path}")