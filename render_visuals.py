"""Compatibility entry point; implementation moved to tools/render_visuals.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('tools.render_visuals', run_name="__main__")
