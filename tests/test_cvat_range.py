import ast
import copy
import tempfile
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from pathlib import Path
from diagnostics.import_cvat_range import import_range
from diagnostics.legacy_baseline import Bind, PREFIX, generate


class RangeTests(unittest.TestCase):
    def fixture(self):
        r=ET.fromstring('<annotations><meta><task><size>5</size><start_frame>0</start_frame>'
            '<stop_frame>4</stop_frame><name>synthetic</name><original_size><width>100</width>'
            '<height>80</height></original_size></task></meta></annotations>')
        t=ET.SubElement(r,'track',id='0',label='person',source='manual')
        for n in range(5):
            ET.SubElement(t,'box',frame=str(n),xtl='0',ytl='0',xbr='10',ybr='20',outside='0',occluded='0')
        timing={'video':'synthetic','video_sha256':'test','average_fps':25,
                'frames':[{'frame':n,'timestamp_seconds':n*.04} for n in range(5)]}
        return r,timing

    def test_boundary_is_exclusive_and_native_ids_are_preserved(self):
        r,t=self.fixture();doc,audit=import_range(r,t,3,'hash')
        self.assertEqual([f['frame'] for f in doc['frames']],[0,1,2])
        self.assertEqual(doc['frames'][0]['items'][0]['person_id'],1)
        self.assertEqual(doc['frames'][0]['items'][0]['annotation_track_id'],'0')
        self.assertEqual(audit['mot_range'],[1,3]);self.assertEqual(doc['annotation_status'],'pending')

    def test_outside_gap_is_not_interpolated(self):
        r,t=self.fixture();track=r.find('track');track.remove(track[2])
        track[1].set('outside','1')
        doc,audit=import_range(r,t,4,'hash')
        self.assertEqual([len(f['items']) for f in doc['frames']],[1,0,0,1])
        self.assertEqual(audit['tracks'][0]['reappearance_frames'],[3])

    def test_missing_visible_frame_and_wrong_review_are_rejected(self):
        r,t=self.fixture();r.find('track').remove(r.find('track')[1])
        self.assertRaises(ValueError,import_range,r,t,3,'hash')
        r,t=self.fixture();self.assertRaises(ValueError,import_range,r,t,3,'hash',{'export_sha256':'wrong'})

    def test_duplicate_native_id_is_rejected(self):
        r,t=self.fixture();r.append(copy.deepcopy(r.find('track')))
        self.assertRaises(ValueError,import_range,r,t,3,'hash')

    def test_asset_binding_changes_no_tracking_overrides(self):
        original='INPUT_VIDEO_PATH = "old"\ntracker = create_tracker(tracker_config=CONFIG_DIR / "old.yaml")\ntracker.track_thresh = 0.25\ntracker.track_buffer = 90\n'
        tree=Bind({'INPUT_VIDEO_PATH':'new'},'actual.yaml').visit(ast.parse(original))
        ast.fix_missing_locations(tree);code=ast.unparse(tree)
        self.assertIn("INPUT_VIDEO_PATH = 'new'",code)
        self.assertIn("Path('actual.yaml')",code)
        self.assertIn('tracker.track_thresh = 0.25',code)
        self.assertIn('tracker.track_buffer = 90',code)
        compile('BASELINE_END=901\n'+PREFIX+code,'synthetic','exec')

    def test_capture_bounds_without_real_video_or_models(self):
        class Capture:
            def __init__(self,path):self.n=0
            def isOpened(self):return True
            def read(self):self.n+=1;return True,object()
            def get(self,prop):return 60
            def release(self):pass
        cv=types.ModuleType('cv2');cv.VideoCapture=Capture;cv.CAP_PROP_FRAME_COUNT=7
        torch=types.ModuleType('torch');torch.cuda=types.SimpleNamespace(is_available=lambda:True)
        namespace={'BASELINE_END':3}
        with patch.dict('sys.modules',{'cv2':cv,'torch':torch}):
            exec(PREFIX,namespace)
            cap=cv.VideoCapture('synthetic')
            self.assertEqual(cap.get(7),3)
            self.assertTrue(all(cap.read()[0] for _ in range(3)))
            self.assertEqual(cap.read(),(False,None))
            self.assertEqual(cap.cap.n,3)

    def test_generated_scripts_compile_and_keep_source_files(self):
        from diagnostics.core import digest
        names=['detect_yolo.py','track.py','stitch_tracks.py']
        before={n:digest(n) for n in names}
        with tempfile.TemporaryDirectory() as tmp:
            c={'pilot':{'video':'synthetic','start_frame':0,'end_frame_exclusive':901},
               'detector':{'checkpoint':'synthetic.pt'},'tracker':{'config_file':'synthetic.yaml'}}
            generate(c,tmp)
            for n in names:
                text=(Path(tmp)/'legacy'/n).read_text()
                compile(text,n,'exec')
                self.assertIn('BASELINE_END = 901',text)
                self.assertIn('Incomplete baseline frame coverage',text)
            self.assertIn('downloads forbidden',(Path(tmp)/'legacy/detect_yolo.py').read_text())
        self.assertEqual(before,{n:digest(n) for n in names})


if __name__=='__main__':unittest.main()
