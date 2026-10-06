"""No-ML checks for relocation, entry points and preserved algorithm ASTs."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from project_paths import ROOT, load_paths

PATH_BINDINGS = {'_paths','MODEL_NAME','INPUT_VIDEO_PATH','INPUT_VIDEO','INPUT_JSON_PATH',
                 'INPUT_JSON','OUTPUT_VIDEO_PATH','OUTPUT_JSON_PATH','OUTPUT_VIDEO','OUTPUT_JSON',
                 'OUTPUT_FOLDER','INPUT_JSON_CANDIDATES','CALIBRATION_FILE','ROBOFLOW_API_KEY'}


def algorithm_fingerprint(text, pipeline=False):
    tree=ast.parse(text)
    main=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'),None)
    if pipeline and main:tree=ast.Module(body=main.body,type_ignores=[])
    class Normalize(ast.NodeTransformer):
        def visit_Import(self,node):return None
        def visit_ImportFrom(self,node):return None
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id in PATH_BINDINGS for t in node.targets):return None
            return self.generic_visit(node)
        def visit_If(self,node):
            if isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='__name__':
                return [self.visit(n) for n in node.body]
            return self.generic_visit(node)
        def visit_keyword(self,node):
            if node.arg=='tracker_config':node.value=ast.Constant('PATH_CONFIGURATION')
            return self.generic_visit(node)
    normalized=Normalize().visit(tree)
    return hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()


class LayoutTests(unittest.TestCase):
    def test_paths_are_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as d:
            config=Path(d)/'paths.json';config.write_text(json.dumps({'video':'videos/example.mp4','output_dir':'outputs/example'}))
            p=load_paths(str(config))
            self.assertEqual(Path(p['video']),ROOT/'videos/example.mp4')
            self.assertEqual(Path(p['calibration']),ROOT/'configs/calibration/calibration_matrix_playground.json')
            config.write_text('{"confidence": 0.99}')
            self.assertRaises(ValueError,load_paths,str(config))

    def test_main_imports_and_help_do_not_load_ml(self):
        for name in ['detect_yolo','track','stitch_tracks','analyze_subgroups']:
            module=importlib.import_module('pipeline.'+name)
            self.assertTrue(callable(module.main))
            with tempfile.TemporaryDirectory() as d:
                for script in [ROOT/(name+'.py'),ROOT/'pipeline'/(name+'.py')]:
                    p=subprocess.run([sys.executable,str(script),'--help'],cwd=d,capture_output=True,text=True)
                    self.assertEqual(p.returncode,0,p.stderr)
                    self.assertIn('--config',p.stdout)

    def test_algorithm_ast_unchanged_except_path_bindings_and_imports(self):
        expected=json.loads((ROOT/'docs/layout_semantics.json').read_text())
        for path,record in expected.items():
            actual=algorithm_fingerprint((ROOT/path).read_text(encoding='utf8'),record['pipeline'])
            self.assertEqual(actual,record['before_hash'],path)

    def test_canonical_paths_and_config_assets_exist(self):
        mapping=json.loads((ROOT/'docs/path_mapping.json').read_text())
        for old,new in mapping.items():
            if new.startswith('assets/'):continue  # private images not distributed
            self.assertTrue((ROOT/new).is_file(),new)
            if old.endswith('.py'):self.assertIn(new,(ROOT/old).read_text())


if __name__=='__main__':unittest.main()
