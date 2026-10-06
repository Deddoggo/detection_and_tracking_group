import numpy as np
import math

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1 = max(x1, x1g); yi1 = max(y1, y1g)
    xi2 = min(x2, x2g); yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_pose_vector(keypoints, conf_thresh=0.5):
    """Compute an ensemble viewing-direction vector:
    1. Shoulder direction (body orientation) - considered the most stable.
    2. Face direction (face orientation) - considered the most detailed."""
    if keypoints is None or len(keypoints) == 0: return None
    
    if hasattr(keypoints, 'cpu'): kpts = keypoints.cpu().numpy()
    else: kpts = keypoints
    
    # Indices: 0:Nose, 1:L-Eye, 2:R-Eye, 3:L-Ear, 4:R-Ear, 5:L-Sh, 6:R-Sh
    nose = kpts[0]
    l_ear = kpts[3]; r_ear = kpts[4]
    l_sh = kpts[5];  r_sh = kpts[6]
    
    vectors = []
    weights = []

    # --- SOURCE 1: SHOULDER NORMAL VECTOR (BODY) ---
    # Line connecting the two shoulders.
    if l_sh[2] > conf_thresh and r_sh[2] > conf_thresh:
        # Shoulder vector: left -> right.
        shoulder_vec = r_sh[:2] - l_sh[:2] 
        # Normal vector (perpendicular): (-y, x) -> points outward from the chest.
        # Note: image coordinates have y increasing downward.
        # The left shoulder appears on the right of the image when facing the camera. Verify this logic empirically.
        # Typical normal vector: (dy, -dx).
        normal_vec = np.array([shoulder_vec[1], -shoulder_vec[0]])
        
        # Normalize.
        norm = np.linalg.norm(normal_vec)
        if norm > 0:
            vectors.append(normal_vec / norm)
            weights.append(1.0) # Body weight: 1.0 (considered highly reliable).

    # --- SOURCE 2: EARS -> NOSE VECTOR (HEAD) ---
    if nose[2] > conf_thresh and l_ear[2] > conf_thresh and r_ear[2] > conf_thresh:
        mid_ear = (l_ear[:2] + r_ear[:2]) / 2
        head_vec = nose[:2] - mid_ear
        
        norm = np.linalg.norm(head_vec)
        if norm > 0:
            vectors.append(head_vec / norm)
            weights.append(1.5) # Head weight: 1.5 (prioritize head direction over body direction).

    # --- COMBINE (WEIGHTED AVERAGE) ---
    if not vectors:
        # Fallback: if neither shoulder nor ear cues are available, try shoulders -> nose as before.
        if nose[2] > conf_thresh and l_sh[2] > conf_thresh and r_sh[2] > conf_thresh:
             mid_sh = (l_sh[:2] + r_sh[:2]) / 2
             fallback_vec = nose[:2] - mid_sh
             return _post_process_vector(fallback_vec)
        return None

    # Compute the weighted average.
    avg_vec = np.average(vectors, axis=0, weights=weights)
    
    return _post_process_vector(avg_vec)

def _post_process_vector(vec):
    """Post-process to counter the 'pointing toward the sky' artifact."""
    # 1. Compress the Y component (perspective correction).
    # Reduce Y by 50% so the vector tends to point more horizontally.
    # Intended to compensate for the elevated camera view.
    vec[1] *= 0.5 
    
    # 2. Normalize again.
    norm = np.linalg.norm(vec)
    if norm < 0.1: return None # Discard vectors that are too short (looking directly at the camera).
    
    return vec / norm

def cosine_similarity(v1, v2):
    if v1 is None or v2 is None: return 0 
    return np.dot(v1, v2)