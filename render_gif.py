"""Compatibility entry point; implementation moved to tools/render_gif.py."""
if __name__ == "__main__":
    import runpy
    runpy.run_module('tools.render_gif', run_name="__main__")
