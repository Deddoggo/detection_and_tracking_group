import os
import cv2
from ultralytics.models.sam import SAM3VideoSemanticPredictor


def segment_video_with_sam3(
    video_path: str,
    prompt_text: str,
    model_path: str = "sam3.pt",
    output_path: str = "output_sam3.mp4",
    conf: float = 0.25,
    imgsz: int = 640,
    half: bool = True,
):
    """
    Segment all instances matching a text prompt in a video using SAM 3 + Ultralytics.

    Args:
        video_path: path to input video
        prompt_text: text prompt, e.g. "person"
        model_path: path to sam3.pt
        output_path: path to output annotated video
        conf: confidence threshold
        imgsz: inference image size
        half: use FP16 if supported by GPU
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SAM3 weights not found: {model_path}\n"
            "Bạn cần tải sam3.pt từ Hugging Face và đặt cùng thư mục hoặc chỉ định full path."
        )

    # Read video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # SAM 3 video semantic predictor
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=model_path,
        imgsz=imgsz,
        half=half,
        verbose=True,
    )
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)

    # Stream results frame-by-frame
    results = predictor(source=video_path, text=[prompt_text], stream=True)

    frame_idx = 0
    for r in results:
        # r.plot() returns frame with masks/boxes/labels drawn
        plotted = r.plot()

        # Ensure output frame size matches original video size
        if plotted.shape[1] != width or plotted.shape[0] != height:
            plotted = cv2.resize(plotted, (width, height))

        writer.write(plotted)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames...")

    writer.release()
    print(f"Done! Saved output video to: {output_path}")


if __name__ == "__main__":
    # Example usage
    segment_video_with_sam3(
        video_path="chunks/video_recess_004.mp4",
        prompt_text="person",
        model_path="sam3.pt",
        output_path="output_person_segmented.mp4",
        conf=0.25,
        imgsz=640,
        half=True,
    )