"""Compatibility entry point; implementation moved to evaluation/eval_tracking_group.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('evaluation.eval_tracking_group', run_name="__main__")
