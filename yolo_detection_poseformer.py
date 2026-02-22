import pickle # --- THÊM THƯ VIỆN NÀY ĐỂ LƯU DỮ LIỆU OFFLINE ---
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

# --- IMPORT HUGGING FACE & BOXMOT ---
from boxmot import create_tracker
from identify_group import cluster_bboxes_with_ids
from orientation_utils import calculate_iou, get_pose_vector, cosine_similarity
try:
    from transformers import AutoProcessor, VitPoseForPoseEstimation
except ImportError:
    print("⚠️ Lỗi: Chưa cài transformers.")

# --- CLASS WRAPPER CHO HUGGING FACE VITPOSE (Giữ nguyên của bạn) ---
class HFViTPosePredictor:
    def __init__(self, model_name="usyd-community/vitpose-base-simple", device='cuda:0'):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def predict(self, img_rgb, bboxes_xyxy):
        if self.model is None or len(bboxes_xyxy) == 0:
            return [None] * len(bboxes_xyxy)
        try:
            boxes_coco = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in bboxes_xyxy]
            inputs = self.processor(img_rgb, boxes=[boxes_coco], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            pose_results = self.processor.post_process_pose_estimation(outputs, boxes=[boxes_coco])[0]
            
            final_keypoints = []
            for res in pose_results:
                kpts = res['keypoints'].cpu().numpy()
                scores = res['scores'].cpu().numpy()
                kpts_with_conf = np.zeros((len(kpts), 3))
                kpts_with_conf[:, :2] = kpts
                kpts_with_conf[:, 2] = scores
                final_keypoints.append(kpts_with_conf)
            return final_keypoints
        except Exception as e:
            return [None] * len(bboxes_xyxy)

# --- PARSE ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--input_video', type=str, required=True)
parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument('--seq_name', type=str, required=True)
parser.add_argument('--calibration_file', type=str, default='calibration_matrix.json')
args = parser.parse_args()
os.makedirs(args.output_folder, exist_ok=True)

# 1. SETUP THIẾT BỊ
device_yolo = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device_boxmot = 0 if torch.cuda.is_available() else 'cpu'

# 2. LOAD MODELS
print("Loading YOLO Detection model & ViTPose...")
model = YOLO('yolo12x.pt').to(device_yolo)
vitpose = HFViTPosePredictor(device=device_yolo)

tracker = create_tracker(
    tracker_type='deepocsort',
    tracker_config=None, 
    reid_weights='osnet_x1_0_msmt17.pt',
    device=device_boxmot,
    half=True
)

cap = cv2.VideoCapture(args.input_video)
frame_idx = 0

# --- LOGIC MỚI: TẠO TỪ ĐIỂN LƯU TRỮ 2D POSE OFFLINE ---
# Cấu trúc: { track_id: { frame_id: np.array(17, 3) } }
offline_2d_pose_data = {}

print(f"Start processing {args.seq_name}...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    current_frame_id = frame_idx + 1
    
    # A. Detect & Track
    results = model.predict(frame, conf=0.4, imgsz=1280, verbose=False, classes=0)[0]
    dets = results.boxes.data.cpu().numpy() 
    tracks = tracker.update(dets, frame) if len(dets) > 0 else tracker.update(np.empty((0, 6)), frame)

    valid_tracks_list = [] 
    bboxes_for_pose = []  

    if len(tracks) > 0:
        valid_tracks = tracks[tracks[:, 4] != -1]
        for trk in valid_tracks:
            bboxes_for_pose.append(trk[:4])
            valid_tracks_list.append(trk)

    # B. Pose Estimation (Trích xuất 2D)
    if len(bboxes_for_pose) > 0:
        pose_results = vitpose.predict(frame_rgb, bboxes_for_pose)
        
        for i, trk in enumerate(valid_tracks_list):
            tid = int(trk[4])
            kpts = pose_results[i] if i < len(pose_results) else None
            
            if kpts is not None:
                # --- LOGIC MỚI: LƯU TỌA ĐỘ VÀO TỪ ĐIỂN ---
                if tid not in offline_2d_pose_data:
                    offline_2d_pose_data[tid] = {}
                offline_2d_pose_data[tid][current_frame_id] = kpts
                
                # (Phần vẽ râu ria hoặc DBSCAN của bạn có thể để ở đây nếu muốn giữ lại video demo)
                # ...

    print(f"Processed frame {current_frame_id}", end='\r')
    frame_idx += 1

cap.release()

# --- LOGIC MỚI: XUẤT FILE DỮ LIỆU OFFLINE ---
pose_output_path = os.path.join(args.output_folder, f"{args.seq_name}_2d_poses.pkl")
with open(pose_output_path, 'wb') as f:
    pickle.dump(offline_2d_pose_data, f)

print(f"\n✅ Đã trích xuất xong 2D Pose cho {len(offline_2d_pose_data)} IDs.")
print(f"📁 Dữ liệu lưu tại: {pose_output_path}")