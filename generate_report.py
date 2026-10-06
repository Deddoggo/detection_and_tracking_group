"""Compatibility entry point; implementation moved to tools/generate_report.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('tools.generate_report', run_name="__main__")
