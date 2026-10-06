"""Compatibility entry point; implementation moved to tools/calibration_headless.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('tools.calibration_headless', run_name="__main__")
