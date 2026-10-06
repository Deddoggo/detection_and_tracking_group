"""Generate bounded working copies of the existing YOLO/ByteTrack/stitch scripts.

Only paths, GPU requirement, explicit local assets and capture bounds change.
Original rounding, confidence gate and post-construction overrides are retained.
"""
import ast
from pathlib import Path
from .core import digest, write


PREFIX = '''
from pathlib import Path as _BaselinePath
import cv2 as _BaselineCV2
import torch as _BaselineTorch
if not _BaselineTorch.cuda.is_available():
    raise RuntimeError("GPU baseline requires CUDA; CPU fallback forbidden")
_BaselineCapture = _BaselineCV2.VideoCapture
class _BoundedBaselineCapture:
    def __init__(self, path):
        self.cap = _BaselineCapture(path)
        if not self.cap.isOpened(): raise RuntimeError("Cannot open baseline video")
        self.read_count = 0
    def read(self):
        if self.read_count >= BASELINE_END: return False, None
        ok, frame = self.cap.read()
        if not ok: raise RuntimeError("Source ended before annotated boundary")
        self.read_count += 1
        return ok, frame
    def get(self, prop):
        if prop == _BaselineCV2.CAP_PROP_FRAME_COUNT: return BASELINE_END
        return self.cap.get(prop)
    def isOpened(self): return self.cap.isOpened()
    def release(self): return self.cap.release()
_BaselineCV2.VideoCapture = _BoundedBaselineCapture
'''


class Bind(ast.NodeTransformer):
    def __init__(self, values, tracker_yaml): self.values=values;self.tracker_yaml=tracker_yaml
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in self.values:
            node.value=ast.Constant(self.values[node.targets[0].id])
        return self.generic_visit(node)
    def visit_Call(self,node):
        if isinstance(node.func,ast.Name) and node.func.id=='create_tracker':
            for arg in node.keywords:
                if arg.arg=='tracker_config':
                    arg.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(self.tracker_yaml)],keywords=[])
        return self.generic_visit(node)


def legacy_tree(path):
    """Read canonical stage body, excluding the path-only CLI adapter."""
    tree = ast.parse(path.read_text(encoding='utf8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    body = [n for n in main.body if not (isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == '_paths' for t in n.targets))]
    return ast.Module(body=body, type_ignores=[])


def generate(config,out):
    out=Path(out);video=config['pilot']['video'];end=config['pilot']['end_frame_exclusive']
    if config['pilot']['start_frame']!=0:raise ValueError('Legacy adapter supports source prefix only')
    checkpoint=config['detector']['checkpoint'];yaml=config['tracker']['config_file']
    legacy=out/'legacy';legacy.mkdir()
    data=out/'legacy_outputs';data.mkdir()
    def path(n):return str((data/n).resolve())
    bindings={
        'detect_yolo.py':{'MODEL_NAME':checkpoint,'INPUT_VIDEO_PATH':video,
                         'OUTPUT_VIDEO_PATH':path('detections.mp4'),'OUTPUT_JSON_PATH':path('detections.json')},
        'track.py':{'INPUT_VIDEO_PATH':video,'INPUT_JSON_PATH':path('detections.json'),
                    'OUTPUT_VIDEO_PATH':path('bytetrack.mp4'),'OUTPUT_JSON_PATH':path('tracks_bytetrack_only.json')},
        'stitch_tracks.py':{'INPUT_VIDEO':video,'INPUT_JSON':path('tracks_bytetrack_only.json'),
                            'OUTPUT_JSON':path('tracks_stitched.json'),'OUTPUT_VIDEO':path('stitched.mp4')}}
    manifest={'policy':'bounded generated copies, original scripts untouched; same legacy rounding/thresholds/runtime overrides',
              'scored_range':[0,end-1],'offline_stitching_uses':'all and only annotated window',
              'detector':'YOLO implementation from detect_yolo.py; detection.py Detectron2 is incompatible with this JSON chain',
              'sources':{},'bindings':bindings,'tracker_yaml':yaml}
    for name,values in bindings.items():
        original=Path(__file__).resolve().parents[1]/"pipeline"/name;tree=Bind(values,yaml).visit(legacy_tree(original))
        ast.fix_missing_locations(tree)
        body=ast.unparse(tree)
        if name=='detect_yolo.py':
            body=body.replace('model.to(DEVICE)', 'model.to(DEVICE)\nif model.names.get(0) != "person": raise RuntimeError("Checkpoint class 0 must be person")')
        result=f'BASELINE_END = {end}\n'+PREFIX+'\n'+body+'\n'
        if name=='detect_yolo.py':
            result=result.replace(PREFIX, PREFIX+f'\nif not _BaselinePath({checkpoint!r}).is_file(): raise FileNotFoundError("Explicit checkpoint missing; downloads forbidden")\n')
            result += ("\nimport json as _BaselineJSON\n"
                       "_BaselinePath(OUTPUT_JSON_PATH + '.effective.json').write_text("
                       "_BaselineJSON.dumps(vars(model.predictor.args), indent=2, default=str), encoding='utf8')\n")
        if name=='track.py':
            result += ("\nimport json as _BaselineJSON\nimport inspect as _BaselineInspect\n"
                       "_BaselinePath(OUTPUT_JSON_PATH + '.effective.json').write_text("
                       "_BaselineJSON.dumps({'implementation': _BaselineInspect.getfile(type(tracker)), "
                       "'scalars': {k:v for k,v in vars(tracker).items() if isinstance(v,(str,int,float,bool)) or v is None}}, "
                       "indent=2, default=str), encoding='utf8')\n")
        result+='\nif frame_idx != BASELINE_END: raise RuntimeError("Incomplete baseline frame coverage")\n'
        target=legacy/name;target.write_text(result,encoding='utf8')
        compile(result,str(target),'exec')
        manifest['sources'][name]={'original_sha256':digest(original),'generated_sha256':digest(target)}
    write(out/'legacy_adapter_manifest.json',manifest)
