"""Compatibility entry point; implementation moved to experiments/yolo_detection.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('experiments.yolo_detection', run_name="__main__")
