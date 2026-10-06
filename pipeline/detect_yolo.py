# Allow both python -m PACKAGE.NAME and direct script execution.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from project_paths import stage_paths

def main():
    _paths = stage_paths('detect_yolo')
    import json
    import os

    import cv2
    import torch
    from ultralytics import YOLO

    # --- 1) Model configuration ---
    MODEL_NAME = _paths["detector_checkpoint"]
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = YOLO(MODEL_NAME)
    model.to(DEVICE)
    print("Model loaded successfully.")

    # --- 2) Input / output configuration ---
    INPUT_VIDEO_PATH = _paths["video"]
    OUTPUT_VIDEO_PATH = str(Path(_paths["output_dir"]) / "output_yolo_detect.mp4")
    OUTPUT_JSON_PATH = str(Path(_paths["output_dir"]) / "detections.json")

    CONF_THRESHOLD = 0.15
    IOU_THRESHOLD = 0.45

    # --- 3) Video initialization ---
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {INPUT_VIDEO_PATH}")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    all_detections = {}
    frame_idx = 0

    print(f"Starting detection on {total_frames} frames with {MODEL_NAME}...")

    # --- 4) Frame loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO and keep only class 0 (person).
        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=[0],
            imgsz=1280,
            verbose=False,
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        frame_detections = []

        # Draw boxes and save [x, y, w, h] detections for tracking.
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)

            frame_detections.append(
                {
                    "bbox": [x1, y1, int(x2 - x1), int(y2 - y1)],
                    "conf": float(score),
                    "label": "person",
                }
            )

            # Overlay confidence label for quick visual QA.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            text = f"{score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        all_detections[f"frame_{frame_idx:05d}"] = frame_detections

        out.write(frame)
        frame_idx += 1

        print(f"Processed: {frame_idx}/{total_frames} frames", end="\r")

    # --- 5) Finalize and save ---
    cap.release()
    out.release()

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(all_detections, f, indent=4)

    print(f"\n\nDone. Video saved to: {OUTPUT_VIDEO_PATH}")
    print(f"Detections JSON saved to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
