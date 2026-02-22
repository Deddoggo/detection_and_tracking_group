import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import torch
from pathlib import Path
import time
from boxmot import create_tracker

# --- PARSE ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--input_video', type=str, required=True)
parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument('--seq_name', type=str, required=True)
args = parser.parse_args()

# Tạo thư mục và mở file TXT
os.makedirs(args.output_folder, exist_ok=True)
mot_output_path = os.path.join(args.output_folder, f"{args.seq_name}.txt")
mot_file = open(mot_output_path, "w")

# --- 1. SETUP THIẾT BỊ ---
if torch.cuda.is_available():
    device_yolo = 'cuda:0'
    device_boxmot = 0
else:
    device_yolo = 'cpu'
    device_boxmot = 'cpu'

# --- 2. LOAD MODELS ---
print(f"Loading YOLO model for {args.seq_name}...")
try:
    model = YOLO('yolo12x.pt') 
except:
    model = YOLO('yolov26x.pt')
model.to(device_yolo)

print(f"Initializing DeepOCSORT Tracker...")
tracker = create_tracker(
    tracker_type='deepocsort',
    tracker_config=None, 
    reid_weights='osnet_x1_0_msmt17.pt', # Dùng chung OSNet cho công bằng
    device=device_boxmot,
    half=True   
)

# --- 3. VIDEO / IMAGE SEQUENCE SETUP ---
cap = cv2.VideoCapture(args.input_video)
frame_idx = 0
start_time = time.time()

# --- 4. MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    current_frame_id = frame_idx + 1

    # A. Detect (Chỉ lấy người - classes=0)
    results = model.predict(frame, conf=0.4, verbose=False, classes=0, half = True)[0]
    dets = results.boxes.data.cpu().numpy() 

    # B. Track
    if len(dets) == 0:
        tracks = tracker.update(np.empty((0, 6)), frame)
    else:
        tracks = tracker.update(dets, frame)

    # C. Ghi file TXT chuẩn MOT
    if len(tracks) > 0:
        valid_tracks = tracks[tracks[:, 4] != -1]
        for trk in valid_tracks:
            x1, y1, x2, y2 = trk[:4]
            tid = int(trk[4])
            conf = float(trk[5])
            
            w = x2 - x1
            h = y2 - y1
            
            mot_file.write(f"{current_frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n")

    print(f"Processing frame {current_frame_id}", end='\r')
    frame_idx += 1

# --- 5. CLEANUP ---
mot_file.close()
cap.release()
total_time = time.time() - start_time

print(f"\n✅ Hoàn thành {args.seq_name}! Lưu tại: {mot_output_path}")
print(f"Tốc độ: {frame_idx / total_time:.2f} FPS")