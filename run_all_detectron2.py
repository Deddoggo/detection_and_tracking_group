"""Compatibility entry point; implementation moved to evaluation/run_all_detectron2.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('evaluation.run_all_detectron2', run_name="__main__")
