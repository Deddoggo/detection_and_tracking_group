import numpy as np
import cv2
import os
import time
import argparse
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.application_util import preprocessing

# ===== CONFIGURATION =====
parser = argparse.ArgumentParser()
parser.add_argument('--input_video', type=str, required=True, help='Path to input video file')
parser.add_argument('--frames_dir', type=str, required=True, help='Directory containing extracted frames')
parser.add_argument('--seq_name', type=str, required=True, help='Name of the MOT sequence (e.g., ADL-Rundle-6)')
# Vẫn giữ 2 biến dưới đây để không bị lỗi nếu lệnh gõ trong terminal của bạn lỡ chứa chúng
parser.add_argument('--epsilon', type=float, default=50.0, help='DBSCAN epsilon value')
parser.add_argument('--threshold_overlap', type=float, default=0.7, help='Overlap threshold')
parser.add_argument('--output_folder', type=str, required=True, help='Path to save the result txt file')
args = parser.parse_args()

frames_dir = args.frames_dir
seq_name = args.seq_name

# Tạo thư mục output và mở file text
# Gán output_dir từ argument thay vì fix cứng
output_dir = args.output_folder
os.makedirs(output_dir, exist_ok=True)

mot_output_path = os.path.join(output_dir, f"{seq_name}.txt")
mot_file = open(mot_output_path, "w")

detection_file = os.path.join(frames_dir, "det.npy")
reid_model_path = "./deep_sort/resources/networks/mars-small128.pb"

# ===== LOAD REID MODEL =====
max_cosine_distance = 0.4
nn_budget = 100
model = gdet.create_box_encoder(reid_model_path, batch_size=32)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# ===== LOAD DETECTIONS & FRAMES =====
detections = np.load(detection_file)
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
total_frames = len(frame_files)

# ===== TRACKING WITH DEEPSORT =====
start_time = time.time()  # Bắt đầu đo thời gian xử lý

for frame_idx, frame_name in enumerate(frame_files):
    # Vẫn PHẢI đọc ảnh vì DeepSORT cần ảnh để trích xuất đặc trưng ngoại hình (ReID)
    frame = cv2.imread(os.path.join(frames_dir, frame_name))
    current_frame_id = frame_idx + 1

    # Lấy detections của frame hiện tại
    frame_dets = detections[detections[:, 0] == current_frame_id]
    bboxes = frame_dets[:, 2:6]  # x, y, w, h
    scores = frame_dets[:, 6]

    if len(bboxes) > 0:
        # Trích xuất đặc trưng ReID
        features = model(frame, bboxes)
        det_objects = [Detection(bbox, score, feat) for bbox, score, feat in zip(bboxes, scores, features)]

        # NMS (Non-Maximum Suppression)
        boxes_nms = np.array([d.tlwh for d in det_objects])
        scores_nms = np.array([d.confidence for d in det_objects])
        indices = preprocessing.non_max_suppression(boxes_nms, 0.8, scores_nms)
        det_objects = [det_objects[i] for i in indices]

        # Cập nhật Tracker
        tracker.predict()
        tracker.update(det_objects)
    else:
        # Nếu frame không có ai, vẫn phải predict để Kalman Filter chạy tiếp
        tracker.predict()

    # LẤY KẾT QUẢ TRACKING VÀ GHI VÀO FILE TXT
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlwh()
        track_id = track.track_id
        
        # Ghi vào file theo chuẩn MOT15: frame, id, x, y, w, h, conf, -1, -1, -1
        mot_file.write(f"{current_frame_id},{track_id},{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f},1.0,-1,-1,-1\n")

    print(f"Tracking frame {current_frame_id}/{total_frames}", end='\r')

end_time = time.time()  # Kết thúc đo thời gian

mot_file.close() # Đóng file sau khi chạy xong

print(f"\n🎉 Tracking completed. Result saved to: {mot_output_path}")
print(f"Processed {total_frames} frames in {end_time - start_time:.2f} seconds.")
print(f"Average processing speed: {total_frames / (end_time - start_time):.2f} FPS.")