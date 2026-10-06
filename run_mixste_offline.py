"""Compatibility entry point; implementation moved to experiments/run_mixste_offline.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('experiments.run_mixste_offline', run_name="__main__")
