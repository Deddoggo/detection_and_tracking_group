"""Compatibility entry point; implementation moved to pipeline/stitch_tracks.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('pipeline.stitch_tracks', run_name="__main__")
