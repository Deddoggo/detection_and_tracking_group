import os
import cv2
import numpy as np
import math
import json
from collections import deque, Counter

from orientation_utils import get_pose_vector

# =====================================================================
# CONFIGURATION
# =====================================================================
INPUT_VIDEO = "videos/WIN_20260326_10_00_11_Pro.mp4"
OUTPUT_FOLDER = "playground_results"
OUTPUT_JSON_FINAL = os.path.join(OUTPUT_FOLDER, "final_sociometrics_data.json")
CALIBRATION_FILE = "calibration_matrix_playground.json"

VITPOSE_MODEL = "usyd-community/vitpose-plus-base"
GENDER_MODEL = "gender-classification-rz62r/3"
ROBOFLOW_API_KEY = "1W6S79RDYF5icaq9dh5i"

SCALE_FACTOR = 1.0
EPSILON = 1.75
MIN_SAMPLES = 2
FRAME_STRIDE = 3

# =====================================================================
# 1) MODEL WRAPPERS
# =====================================================================
def validate_vitpose_plus_config(config):
    """Reject a different architecture or joint order rather than mislabeling a cache."""
    from observation_cache import COCO17
    def normalize(label):
        label = label.lower().replace(' ', '_')
        return 'left_'+label[2:] if label.startswith('l_') else 'right_'+label[2:] if label.startswith('r_') else label
    labels = [normalize(config.id2label.get(i, '')) for i in range(17)]
    if (config.model_type != 'vitpose' or labels != COCO17 or config.num_labels != 17
            or config.backbone_config.num_experts != 6 or config.backbone_config.part_features != 192
            or config.use_simple_decoder):
        raise ValueError('expected ViTPose+ Base with six experts and COCO17 ordering')


class HFViTPosePredictor:
    """The single authorized HF model, resolved once to an immutable Hub commit."""
    def __init__(self, device='cuda:0', *, strict=False, revision=None):
        self.strict = strict
        self.last_diagnostics = None
        print(f"Initializing {VITPOSE_MODEL} on {device}...")
        try:
            import inspect
            import re
            from huggingface_hub import HfApi, try_to_load_from_cache
            from transformers import AutoConfig, AutoProcessor, VitPoseForPoseEstimation
            from annotation_io import file_sha256
            if 'dataset_index' not in inspect.signature(VitPoseForPoseEstimation.forward).parameters:
                raise RuntimeError('installed Transformers lacks ViTPose+ dataset_index support')
            if revision is not None and not re.fullmatch(r'[0-9a-f]{40}', revision):
                raise ValueError('revision must be an exact 40-character HF commit SHA')
            resolved = revision or HfApi().model_info(VITPOSE_MODEL, revision='main').sha
            if not re.fullmatch(r'[0-9a-f]{40}', resolved):
                raise ValueError('Hub did not resolve an immutable commit')
            self.device = device
            config = AutoConfig.from_pretrained(VITPOSE_MODEL, revision=resolved, trust_remote_code=False)
            validate_vitpose_plus_config(config)
            self.processor = AutoProcessor.from_pretrained(VITPOSE_MODEL, revision=resolved, trust_remote_code=False)
            self.model = VitPoseForPoseEstimation.from_pretrained(
                VITPOSE_MODEL, revision=resolved, trust_remote_code=False, use_safetensors=True).to(device)
            self.model.eval()
            # Transformers versions differ in private config attributes. Pin the
            # public from_pretrained revision and audit the artifacts at that SHA.
            validate_vitpose_plus_config(self.model.config)
            files = {}
            for name in ('config.json','preprocessor_config.json','model.safetensors'):
                cached = try_to_load_from_cache(VITPOSE_MODEL,name,revision=resolved)
                if not isinstance(cached,str):
                    raise ValueError(f'loaded Hub artifact missing from cache: {name}')
                files[name] = file_sha256(cached)
            self.identity = {'model_id':VITPOSE_MODEL,'resolved_revision':resolved,
                'implementation':'transformers.VitPoseForPoseEstimation','dataset_index':0,
                'dataset':'COCO human pose','joint_order':'COCO17',
                'configured_joint_labels':config.id2label,'num_experts':config.backbone_config.num_experts,
                'files_sha256':files,'loading':'from_pretrained(model_id, revision=resolved_sha)',
                'processor_class':type(self.processor).__name__,'dtype':str(self.model.dtype)}
            print(f'Loaded {VITPOSE_MODEL}@{resolved}')
        except Exception as e:
            if strict:
                raise RuntimeError('ViTPose+ initialization failed; cache export stopped') from e
            print(f'ViTPose+ initialization failed: {e}')
            self.model = None

    def predict(self, img_rgb, bboxes_xyxy):
        if len(bboxes_xyxy) == 0:
            return []
        if self.model is None:
            if self.strict:
                raise RuntimeError('ViTPose+ model unavailable')
            return [None] * len(bboxes_xyxy)
        try:
            import time
            import torch
            boxes_coco = [[x1,y1,x2-x1,y2-y1] for x1,y1,x2,y2 in bboxes_xyxy]
            started = time.perf_counter()
            inputs = self.processor(img_rgb, boxes=[boxes_coco], return_tensors='pt').to(self.device)
            batch = inputs['pixel_values'].shape[0]
            if batch != len(boxes_coco):
                raise ValueError('processor/person count mismatch')
            dataset_index = torch.zeros(batch,dtype=torch.long,device=self.device)
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize(self.device)
            forward_start = time.perf_counter()
            with torch.inference_mode():
                outputs = self.model(**inputs,dataset_index=dataset_index)
            if str(self.device).startswith('cuda'):
                torch.cuda.synchronize(self.device)
            forward_end = time.perf_counter()
            if tuple(outputs.heatmaps.shape[:2]) != (batch,17) or not torch.isfinite(outputs.heatmaps).all().item():
                raise ValueError('invalid ViTPose+ heatmap shape/values')
            pose_results = self.processor.post_process_pose_estimation(outputs,boxes=[boxes_coco])[0]
            if len(pose_results) != len(boxes_coco):
                raise ValueError('pose/person count mismatch')
            final_keypoints = []
            returned_bboxes = []
            for res in pose_results:
                kpts = res['keypoints'].cpu().numpy()
                scores = res['scores'].cpu().numpy()
                labels = res['labels'].cpu().numpy()
                returned_bbox = res['bbox'].cpu().numpy()
                if (kpts.shape != (17,2) or scores.shape != (17,)
                        or not np.array_equal(labels,np.arange(17))
                        or returned_bbox.shape != (4,) or not np.isfinite(returned_bbox).all()):
                    raise ValueError('keypoint order/shape or returned metadata mismatch')
                # In installed Transformers 5.22.2 this bbox is built from center/scale,
                # not the original ROI. Preserve source boxes; associate by input crop
                # order/count (one person per call in the real smoke test).
                returned_bboxes.append(returned_bbox.tolist())
                kpts_with_conf = np.column_stack([kpts,scores])
                if not np.isfinite(kpts_with_conf).all():
                    raise ValueError('nonfinite ViTPose+ result')
                final_keypoints.append(kpts_with_conf)
            self.last_diagnostics = {'input_shape':list(inputs['pixel_values'].shape),
                'heatmap_shape':list(outputs.heatmaps.shape),'output_shape':[batch,17,3],
                'dataset_index':dataset_index.cpu().tolist(),'labels_and_positional_association_verified':True,
                'hf_returned_bboxes':returned_bboxes,'hf_returned_bbox_used_for_association':False,
                'preprocess_seconds':forward_start-started,'forward_seconds':forward_end-forward_start,
                'postprocess_seconds':time.perf_counter()-forward_end}
            return final_keypoints
        except Exception as e:
            if self.strict:
                raise RuntimeError('ViTPose+ inference failed; cache remains incomplete') from e
            return [None] * len(bboxes_xyxy)

class RoboflowGenderPredictor:
    def __init__(self, model_id=GENDER_MODEL, api_key=ROBOFLOW_API_KEY):
        print(f"Initializing Roboflow Gender Classification ({model_id})...")
        try:
            from inference import get_model
            self.model = get_model(model_id=model_id, api_key=api_key)
            print("✅ Roboflow Gender Classification initialized successfully.")
        except Exception as e:
            print(f"❌ Error Roboflow Model: {e}")
            self.model = None

    def predict(self, crops_bgr):
        if self.model is None or len(crops_bgr) == 0:
            return ["Unknown"] * len(crops_bgr)
        try:
            # 1. BẮT BUỘC CHUYỂN MÀU: Đảo BGR của OpenCV về RGB chuẩn của mô hình
            crops_rgb = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops_bgr]
            
            # 2. Gọi model infer với mảng ảnh RGB
            results = self.model.infer(crops_rgb)
            
            predictions = []
            for res in results:
                # 3. LẤY KẾT QUẢ ĐÚNG: Sử dụng thuộc tính 'top' thay vì 'predictions[0]'
                # SDK của Roboflow có thể trả về object (hasattr) hoặc dictionary (isinstance)
                if hasattr(res, 'top') and res.top:
                    predictions.append(res.top)
                elif isinstance(res, dict) and 'top' in res:
                    predictions.append(res['top'])
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
    def __init__(self, tracks_path, *, compatibility_b0=False):
        if not compatibility_b0:
            raise ValueError("live path requires explicit compatibility_b0=True; use C2 replay for verified units")
        import torch
        self.tracks_path = tracks_path
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

        self.vitpose = HFViTPosePredictor(device=self.device)
        self.gender_predictor = RoboflowGenderPredictor(model_id=GENDER_MODEL, api_key=ROBOFLOW_API_KEY)

    def _load_calibration(self):
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, 'r') as f:
                return np.array(json.load(f)['homography_matrix'], dtype=np.float32)
        return None

    def _load_tracks_json(self):
        # C2 requires explicit raw/stitched selection. Inference behavior is unchanged.
        with open(self.tracks_path, 'r', encoding='utf8') as f:
            return json.load(f)

    def process_frame(self, frame, frame_idx):
        from identify_group_social import cluster_bboxes_with_ids
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
            scale_factor=SCALE_FACTOR, eps=EPSILON, min_samples=MIN_SAMPLES,
            geometry_mode="legacy-b0"
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

def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('validate-data', 'cache', 'replay'):
        from c2_pipeline import main as c2_main
        return c2_main(argv)
    import argparse
    parser = argparse.ArgumentParser(description='Live B0 compatibility or C2 validate-data/cache/replay')
    parser.add_argument('--compatibility-b0', action='store_true', required=True,
                        help='Explicitly retain unverified historical calibration/unit semantics')
    parser.add_argument('--tracks', required=True, help='Explicit raw or stitched tracks JSON')
    args = parser.parse_args(argv)
    SubgroupAnalyzerPipeline(args.tracks, compatibility_b0=args.compatibility_b0).run()


if __name__ == "__main__":
    main()
