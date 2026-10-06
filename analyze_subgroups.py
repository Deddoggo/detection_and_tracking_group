"""Compatibility entry point; implementation moved to pipeline/analyze_subgroups.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('pipeline.analyze_subgroups', run_name="__main__")
