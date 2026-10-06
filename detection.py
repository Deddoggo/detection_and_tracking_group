"""Compatibility entry point; implementation moved to legacy/detection.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('legacy.detection', run_name="__main__")
