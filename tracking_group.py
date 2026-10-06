"""Compatibility entry point; implementation moved to legacy/tracking_group.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('legacy.tracking_group', run_name="__main__")
