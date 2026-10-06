"""Compatibility entry point; implementation moved to tools/measure_tool_headless.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('tools.measure_tool_headless', run_name="__main__")
