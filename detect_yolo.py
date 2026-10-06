"""Compatibility entry point; implementation moved to pipeline/detect_yolo.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('pipeline.detect_yolo', run_name="__main__")
