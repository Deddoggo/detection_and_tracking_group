"""Compatibility entry point; implementation moved to evaluation/tracking_yolov12.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('evaluation.tracking_yolov12', run_name="__main__")
