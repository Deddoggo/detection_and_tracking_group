"""Compatibility entry point; implementation moved to tools/calibration.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('tools.calibration', run_name="__main__")
