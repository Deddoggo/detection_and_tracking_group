"""Compatibility entry point; implementation moved to pipeline/track.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('pipeline.track', run_name="__main__")
