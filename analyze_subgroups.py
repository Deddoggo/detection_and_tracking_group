import os
import cv2
import numpy as np
import math
import json
from collections import deque, Counter
import torch

from transformers import AutoProcessor, VitPoseForPoseEstimation
from inference import get_model

from identify_group_social import cluster_bboxes_with_ids
from orientation_utils import get_pose_vector

# =====================================================================
# CONFIGURATION
# =====================================================================
INPUT_VIDEO = "videos/WIN_20260326_10_00_11_Pro.mp4"
INPUT_JSON_CANDIDATES = [
    "playground_results/tracks_stitched.json",
    "playground_results/tracks_bytetrack_only.json",
]
OUTPUT_FOLDER = "playground_results"
OUTPUT_JSON_FINAL = os.path.join(OUTPUT_FOLDER, "final_sociometrics_data.json")
CALIBRATION_FILE = "calibration_matrix_playground.json"

VITPOSE_MODEL = "usyd-community/vitpose-base-simple"
GENDER_MODEL = "gender-classification-rz62r/3"
ROBOFLOW_API_KEY = "1W6S79RDYF5icaq9dh5i"

SCALE_FACTOR = 1.0
EPSILON = 1.75
MIN_SAMPLES = 2
FRAME_STRIDE = 3

# =====================================================================
# 1) MODEL WRAPPERS
# =====================================================================
class HFViTPosePredictor:
    def __init__(self, model_name=VITPOSE_MODEL, device='cuda:0'):
        print(f"Initializing HF ViTPose ({model_name})...")
        try:
            self.device = device
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = VitPoseForPoseEstimation.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.model.eval()
            print("✅ HF ViTPose initialized successfully.")
        except Exception as e:
            print(f"❌ Error ViTPose: {e}")
            self.model = None

    def predict(self, img_rgb, bboxes_xyxy):
        if self.model is None or len(bboxes_xyxy) == 0: return [None] * len(bboxes_xyxy)
        try:
            boxes_coco = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in bboxes_xyxy]
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

class RoboflowGenderPredictor:
    def __init__(self, model_id=GENDER_MODEL, api_key=ROBOFLOW_API_KEY):
        print(f"Initializing Roboflow Gender Classification ({model_id})...")
        try:
            self.model = get_model(model_id=model_id, api_key=api_key)
            print("✅ Roboflow Gender Classification initialized successfully.")
        except Exception as e:
            print(f"❌ Error Roboflow Model: {e}")
            self.model = None

    def predict(self, crops_bgr):
        if self.model is None or len(crops_bgr) == 0:
            return ["Unknown"] * len(crops_bgr)
        try:
            # Truyền nguyên 1 batch (list các numpy array ảnh BGR) để tận dụng GPU
            results = self.model.infer(crops_bgr)
            
            predictions = []
            for res in results:
                if hasattr(res, 'predictions') and len(res.predictions) > 0:
                    predictions.append(res.predictions[0].class_name)
                else:
                    predictions.append("Unknown")
            return predictions
            
        except Exception as e:
            print(f"⚠️ Gender Prediction Error: {e}")
            return ["Unknown"] * len(crops_bgr)

# =====================================================================
# 2) CORE PIPELINE (HEADLESS)
# =====================================================================
class SubgroupAnalyzerPipeline:
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.track_history = {}
        self.pose_history_smooth = {}
        self.groups_status = {}
        self.max_group_id = -1
        self.gender_history = {}
        self.gender_consensus = {}
        self.ALPHA_SMOOTH = 0.7
        
        self.homography_matrix = self._load_calibration()
        self.tracks_data = self._load_tracks_json()

        self.vitpose = HFViTPosePredictor(model_name=VITPOSE_MODEL, device=self.device)
        self.gender_predictor = RoboflowGenderPredictor(model_id=GENDER_MODEL, api_key=ROBOFLOW_API_KEY)

    def _load_calibration(self):
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, 'r') as f:
                return np.array(json.load(f)['homography_matrix'], dtype=np.float32)
        return None

    def _load_tracks_json(self):
        for candidate in INPUT_JSON_CANDIDATES:
            if os.path.exists(candidate):
                print(f"Using tracking input: {candidate}")
                with open(candidate, 'r') as f:
                    return json.load(f)
        raise FileNotFoundError(
            f"No tracking file found. Checked: {', '.join(INPUT_JSON_CANDIDATES)}"
        )

    def process_frame(self, frame, frame_idx):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_key = f"frame_{frame_idx:05d}"
        frame_tracks = self.tracks_data.get(frame_key, [])
        
        bboxes_for_pose, boxes_xywh, ids = [], [], []
        current_frame_ids = set()

        for trk in frame_tracks:
            tid = trk['id']
            x, y, w, h = trk['bbox']

            # Skip tiny boxes to reduce false positives and unstable pose predictions.
            if w < 15 or h < 30:
                continue

            bboxes_for_pose.append([x, y, x + w, y + h]) 
            boxes_xywh.append([x, y, w, h])              
            ids.append(tid)
            current_frame_ids.add(tid)

        self.pose_history_smooth = {k: v for k, v in self.pose_history_smooth.items() if k in current_frame_ids}
        self.gender_history = {k: v for k, v in self.gender_history.items() if k in current_frame_ids}
        self.gender_consensus = {k: v for k, v in self.gender_consensus.items() if k in current_frame_ids}

        # Continuous gender voting over recent frames improves stability.
        valid_crops, valid_crop_indices = [], []
        for i, (x, y, w, h) in enumerate(boxes_xywh):
            tid = ids[i]

            # Extract person crop from the ORIGINAL BGR frame directly
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                valid_crops.append(crop)
                valid_crop_indices.append(i)

        if len(valid_crops) > 0:
            gender_preds = self.gender_predictor.predict(valid_crops)
            for idx, pred in zip(valid_crop_indices, gender_preds):
                tid = ids[idx]
                if tid not in self.gender_history:
                    self.gender_history[tid] = deque(maxlen=50) 
                self.gender_history[tid].append(pred)
                self.gender_consensus[tid] = Counter(self.gender_history[tid]).most_common(1)[0][0]

        # --- ViTPose + temporal smoothing ---
        active_tracks_info = {}
        pose_vectors_list = []

        if len(bboxes_for_pose) > 0:
            pose_results = self.vitpose.predict(frame_rgb, bboxes_for_pose)
            for i, tid in enumerate(ids):
                kpts = pose_results[i] if i < len(pose_results) else None
                raw_pose_vec = get_pose_vector(kpts, conf_thresh=0.6)
                
                if raw_pose_vec is not None:
                    if tid in self.pose_history_smooth:
                        old_vx, old_vy = self.pose_history_smooth[tid]
                        new_vx = self.ALPHA_SMOOTH * raw_pose_vec[0] + (1 - self.ALPHA_SMOOTH) * old_vx
                        new_vy = self.ALPHA_SMOOTH * raw_pose_vec[1] + (1 - self.ALPHA_SMOOTH) * old_vy
                        norm = math.sqrt(new_vx**2 + new_vy**2)
                        final_pose_vec = (new_vx/norm, new_vy/norm) if norm > 0 else raw_pose_vec
                    else:
                        final_pose_vec = raw_pose_vec
                    self.pose_history_smooth[tid] = final_pose_vec
                else:
                    final_pose_vec = self.pose_history_smooth.get(tid, None)

                pose_vectors_list.append(final_pose_vec)
                active_tracks_info[tid] = {'gender': self.gender_consensus.get(tid, "Unknown")}

        # --- Subgroup clustering ---
        cluster_results, self.groups_status, self.max_group_id = cluster_bboxes_with_ids(
            self.groups_status, boxes_xywh, ids, self.max_group_id, 
            pose_vectors=pose_vectors_list, homography_matrix=self.homography_matrix, 
            scale_factor=SCALE_FACTOR, eps=EPSILON, min_samples=MIN_SAMPLES
        )

        return cluster_results, active_tracks_info

    def run(self):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        cap = cv2.VideoCapture(INPUT_VIDEO)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_frames_data = {}
        frame_idx = 0
        print(f"🚀 Starting AI analysis on {total_frames} frames (fast stride mode)...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip intermediate frames for speed.
            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                continue

            # Run AI only on selected stride frames.
            cluster_results, active_tracks_info = self.process_frame(frame, frame_idx)

            # Package output records for JSON serialization.
            frame_data = []
            for person in cluster_results:
                tid = int(person['id_p'])
                real_pos = person.get('real_pos')
                clean_bbox = [float(x) for x in person['bbox']]

                frame_data.append({
                    'id_p': tid,
                    'id_g': int(person['id_g']),
                    'bbox': clean_bbox,
                    'gender': active_tracks_info.get(tid, {}).get('gender', 'Unknown'),
                    'real_pos': real_pos.tolist() if real_pos is not None else None 
                })
            
            all_frames_data[f"frame_{frame_idx:05d}"] = frame_data
            
            frame_idx += 1
            print(f"Progress (stride={FRAME_STRIDE}): {frame_idx}/{total_frames}", end='\r')

        cap.release()
        with open(OUTPUT_JSON_FINAL, 'w') as f:
            json.dump(all_frames_data, f, indent=4)
        print(f"\nDone. Sociometrics output saved to: {OUTPUT_JSON_FINAL}")

if __name__ == "__main__":
    SubgroupAnalyzerPipeline().run()