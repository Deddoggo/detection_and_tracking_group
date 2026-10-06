"""Compatibility entry point; implementation moved to tools/measure_tool.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('tools.measure_tool', run_name="__main__")
