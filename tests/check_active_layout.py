"""Lightweight source checks: python -B tests/check_active_layout.py."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
ACTIVE = {'detect_yolo.py', 'track.py', 'stitch_tracks.py', 'analyze_subgroups.py',
          'identify_group_social.py', 'orientation_utils.py', 'calibration_headless.py',
          'render_visuals.py', 'generate_report.py'}
HELPERS = {'group_identity.py', 'orientation_gate.py', 'grouping_contracts.py', 'report_metrics.py',
           'annotation_io.py', 'observation_cache.py', 'c2_pipeline.py'}


class ActiveLayout(unittest.TestCase):
    def test_active_sources_compile_without_import_side_effects(self):
        for name in ACTIVE:
            with self.subTest(source=name):
                text = (ROOT / name).read_text(encoding='utf-8-sig')
                compile(text, name, 'exec')  # Compilation only: no model initialization.

    def test_root_source_boundary_and_local_imports(self):
        self.assertLessEqual({p.name for p in ROOT.glob('*.py')}, ACTIVE | HELPERS)
        for name in ACTIVE | {p.name for p in ROOT.glob('*.py') if p.name in HELPERS}:
            tree = ast.parse((ROOT / name).read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    local = node.module + '.py'
                    if local in ACTIVE | HELPERS:
                        self.assertTrue((ROOT / local).is_file(), f'{name} requires {local}')

    def test_required_calibration_resources(self):
        self.assertTrue((ROOT / 'Real_playground.png').is_file())
        self.assertTrue((ROOT / 'calibration_matrix_playground.json').is_file())
        for p in ROOT.glob('calibration_matrix*.json'):
            self.assertGreater(p.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
