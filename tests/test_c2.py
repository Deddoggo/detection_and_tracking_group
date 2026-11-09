"""C2 contracts/native replay. Synthetic poses are explicitly test-only."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
import cv2
from annotation_io import read_cvat, verify_decode
from observation_cache import (SCHEMA,COCO17,CacheWriter,canonical_hash,read_cache,
                               validate_frame,validate_manifest)
from c2_pipeline import GroupingReplay,observation_frame
from grouping_contracts import geometry_contract,project_geometry


def manifest(end=7):
    return {'schema':SCHEMA,'joint_order':COCO17,'video_sha256':'a'*64,'input_sha256':'b'*64,
        'width':1920,'height':1080,'stride':3,'start_frame':0,'end_frame_exclusive':end,
        'time_base':'1/30000','coordinate_units':'pixels','bbox_format':'xywh',
        'id_namespace':'synthetic-test','auxiliary_attributes':'disabled','inference_kind':'synthetic-test',
        'pose_model':{'id':'SYNTHETIC TEST FIXTURE; not model inference'},'provenance':{'test':True}}


def person(tid=1, x=0, w=20, outside=False):
    return {'person_id':tid,'bbox_xywh':[x,100,w,40],'outside':outside,'occluded':False,
            'keyframe':True,'attributes':{'social_group':'unreviewed','gender':'unknown'}}


def keypoints(reverse=False):
    k = np.zeros((17,3),dtype=float)
    k[:,2] = .9
    k[0,:2] = [10,5]
    k[3,:2],k[4,:2] = [6,10],[14,10]
    k[5,:2],k[6,:2] = ([15,20],[5,20]) if reverse else ([5,20],[15,20])
    return k.tolist()


def frame(index=0, people=None, poses=None):
    people = [person(1),person(2,30)] if people is None else people
    poses = [keypoints() for p in people if not p['outside'] and p['bbox_xywh'][2]>=15] if poses is None else poses
    return observation_frame(index,index*1001,'1/30000',people,poses)


class AnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT/'WIN_20260303_10_09_56_PRO_GT_First60s/annotations.xml'
        if cls.path.exists():
            cls.meta,cls.frames = read_cvat(cls.path)

    def test_real_xml_identity_and_coverage(self):
        if not self.path.exists(): self.skipTest('local XML absent')
        self.assertEqual((self.meta['start_frame'],self.meta['stop_frame'],self.meta['width'],self.meta['height']),(0,2955,1920,1080))
        self.assertEqual(len({p['person_id'] for ps in self.frames.values() for p in ps}),76)
        self.assertEqual(len(self.frames),2956)

    def test_outside_interval_no_interpolation(self):
        if not self.path.exists(): self.skipTest('local XML absent')
        self.assertTrue(next(p for p in self.frames[319] if p['person_id']==40)['outside'])
        self.assertFalse(any(p['person_id']==40 for f in range(320,334) for p in self.frames[f]))
        self.assertFalse(next(p for p in self.frames[334] if p['person_id']==40)['outside'])

    def test_partial_social_labels_preserved(self):
        if not self.path.exists(): self.skipTest('local XML absent')
        people = [p for f in range(1800) for p in self.frames[f] if not p['outside']]
        self.assertEqual(len(people),71157)
        self.assertEqual(sum(p['attributes']['social_group']=='unreviewed' for p in people),66223)
        self.assertEqual({p['attributes']['social_group'] for p in people},{'unreviewed',*[f'G{i}' for i in range(1,10)]})

    def test_fractional_boxes_flags_not_modified(self):
        p=person();p['occluded']=True;p['keyframe']=False;p['bbox_xywh'][0]=10.125;p['attributes']['social_group']='G9'
        original=copy.deepcopy(p)
        row=frame(people=[p],poses=[None])
        self.assertEqual(p,original)
        self.assertEqual(row['persons'][0]['reference'],{k:p[k] for k in ('outside','occluded','keyframe','attributes')})
        self.assertEqual(row['persons'][0]['bbox_xywh'][0],10.125)

    def test_bad_xml_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.xml';p.write_text('<annotations><version>1.1</version><meta/></annotations>')
            with self.assertRaises(ValueError):read_cvat(p)

    def test_decoder_rejects_wrong_pts(self):
        image=np.zeros((10,20,3),dtype=np.uint8)
        cap=unittest.mock.Mock()
        cap.read.return_value=(True,image);cap.get.return_value=1000
        with patch('cv2.VideoCapture',return_value=cap):
            with self.assertRaisesRegex(ValueError,'PTS mismatch'):
                verify_decode('not-a-real-video',{'width':20,'height':10,'frame_count':1,'pts_ticks':[0],'time_base':'1/30000'},1)
        cap.release.assert_called_once()


class SchemaTests(unittest.TestCase):
    def test_roundtrip_raw_scores_and_unreviewed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache';w=CacheWriter(p,manifest())
            rows=[frame(i) for i in (0,3,6)]
            for r in rows:w.append(r)
            w.close(complete=True)
            m,result=read_cache(p)
            self.assertEqual(rows,result)
            self.assertEqual(result[0]['persons'][0]['reference']['attributes']['social_group'],'unreviewed')
            self.assertEqual(m['inference_kind'],'synthetic-test')
            self.assertEqual(result[0]['persons'][0]['keypoints_xy_score'][0][2],.9)
            with self.assertRaises(FileExistsError):CacheWriter(p,manifest())

    def test_excluded_and_missing_persons_retained(self):
        r=frame(people=[person(1,w=14),person(2,outside=True),person(3)],poses=[None])
        validate_frame(r,manifest())
        self.assertEqual([p['pose_status'] for p in r['persons']],['excluded','excluded','missing'])
        self.assertEqual([p['exclusion_reason'] for p in r['persons']],['tiny_bbox','outside',None])
        self.assertTrue(all(p['detection_confidence'] is None for p in r['persons']))

    def test_empty_frame_is_valid(self):
        validate_frame(frame(people=[],poses=[]),manifest())

    def test_shape_nonfinite_duplicate_and_pts_errors(self):
        mutations=[lambda r:r['persons'].append(copy.deepcopy(r['persons'][0])),
                   lambda r:r['persons'][0]['keypoints_xy_score'].pop(),
                   lambda r:r['persons'][0]['bbox_xywh'].__setitem__(0,float('nan')),
                   lambda r:r.__setitem__('timestamp_seconds',.1),
                   lambda r:r['persons'][0].__setitem__('eligible',False),
                   lambda r:r['persons'][0]['keypoints_xy_score'][0].__setitem__(2,float('inf'))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                r=frame();mutate(r)
                with self.assertRaises(ValueError):validate_frame(r,manifest())

    def test_invalid_manifest_and_joint_order(self):
        for key,value in [('joint_order',['wrong']),('stride',0),('input_sha256','bad'),('coordinate_units','m'),('inference_kind','guessed')]:
            m=manifest();m[key]=value
            with self.assertRaises(ValueError):validate_manifest(m)

    def test_incomplete_missing_order_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache';w=CacheWriter(p,manifest());w.append(frame())
            with self.assertRaises(ValueError):w.append(frame(6))
            w.close()
            with self.assertRaises(ValueError):read_cache(p)
            p2=Path(d)/'complete';w=CacheWriter(p2,manifest(1));w.append(frame());w.close(complete=True)
            with self.assertRaises(ValueError):read_cache(p2,expected_video_sha256='c'*64)
            with (p2/'frames.jsonl').open('a') as f:f.write('{}\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):read_cache(p2)

    def test_mismatched_pose_person_count_rejected(self):
        with self.assertRaises(ValueError):frame(poses=[])

    def test_vitpose_plus_config_requires_coco17_and_six_experts(self):
        from types import SimpleNamespace
        from analyze_subgroups import validate_vitpose_plus_config
        config=SimpleNamespace(model_type='vitpose',id2label=dict(enumerate(COCO17)),num_labels=17,
            backbone_config=SimpleNamespace(num_experts=6,part_features=192),use_simple_decoder=False)
        validate_vitpose_plus_config(config)
        config.id2label[0]='wrong_joint'
        with self.assertRaises(ValueError):validate_vitpose_plus_config(config)
        config.id2label[0]='nose';config.backbone_config.num_experts=1
        with self.assertRaises(ValueError):validate_vitpose_plus_config(config)

    def test_revision_rejects_mutable_or_local_checkpoint_inputs(self):
        from analyze_subgroups import HFViTPosePredictor
        for revision in ('main','LOCAL_CHECKPOINT_DIRECTORY'):
            with self.assertRaises(RuntimeError):HFViTPosePredictor(device='cpu',strict=True,revision=revision)

    def test_raw_heatmap_score_not_clipped_or_treated_as_probability(self):
        k=keypoints();k[0][2]=1.25
        r=frame(people=[person()],poses=[k]);validate_frame(r,manifest())
        self.assertEqual(r['persons'][0]['keypoints_xy_score'][0][2],1.25)


class GeometryTests(unittest.TestCase):
    def calibration(self,m,H):
        return {'schema':'calibration.v1','calibration_id':'SYNTHETIC-TEST-NOT-FIELD-CALIBRATION',
            'verified':True,'verified_by':'synthetic unit test','units':'m','measurement_evidence':'synthetic fixture',
            'correspondence_tolerance_m':1e-6,
            **{k:m[k] for k in ('video_sha256','width','height')},'homography_matrix':H.tolist(),
            'homography_sha256':canonical_hash(H.tolist()),
            'correspondences':[{'image_xy':[x,y],'ground_xy':[x,y]} for x,y in [(0,0),(1,0),(1,1),(0,1)]]}

    def test_metric_missing_unverified_and_wrong_video_rejected(self):
        from identify_group_social import cluster_bboxes_with_ids
        with self.assertRaises(ValueError):cluster_bboxes_with_ids({},[],[],-1)
        m=manifest();H=np.eye(3);c=self.calibration(m,H)
        for changed in ({'verified':False},{'video_sha256':'c'*64},{'width':100},{'homography_sha256':'bad'},{'units':'pixels'}):
            bad=dict(c,**changed)
            with self.assertRaises(ValueError):geometry_contract('metric',H,calibration=bad,video=m)
        bad=copy.deepcopy(c);bad['correspondences'][0]['ground_xy']=[100,100]
        with self.assertRaisesRegex(ValueError,'ground correspondences'):geometry_contract('metric',H,calibration=bad,video=m)

    def test_synthetic_metric_identity_and_projection(self):
        m=manifest();H=np.eye(3);c=self.calibration(m,H)
        self.assertEqual(geometry_contract('metric',H,calibration=c,video=m)['units'],'m')
        X,v=project_geometry([[0,0,20,40]],[[1.,0]],H)
        np.testing.assert_array_equal(X,[[10,40]])
        np.testing.assert_array_equal(v,[[1,0]])

    def test_image_mode_requires_explicit_pixel_policy(self):
        with self.assertRaises(ValueError):geometry_contract('image-space-diagnostic')
        with self.assertRaises(ValueError):geometry_contract('image-space-diagnostic',np.eye(3),close_distance=10)
        g=geometry_contract('image-space-diagnostic',close_distance=10)
        self.assertEqual(g['units'],'pixels')
        X,v=project_geometry([[0,0,20,40]],[[1,0]],None,image_directions=True)
        self.assertEqual(v,[[1,0]])

    def test_native_chain_and_split_preserve_c1(self):
        from identify_group_social import cluster_bboxes_with_ids
        r,_,_=cluster_bboxes_with_ids({},[[x,0,.1,.1] for x in (0,1,2)],[1,2,3],-1,eps=1.75,geometry_mode='legacy-b0')
        self.assertEqual([p['id_g'] for p in r],[0,0,0])
        r,_,mx=cluster_bboxes_with_ids({7:[1,2,3,4]},[[x,0,.1,.1] for x in (0,1,10,11,50)],[1,2,3,4,5],7,eps=1.75,geometry_mode='legacy-b0')
        self.assertEqual([p['id_g'] for p in r],[7,7,8,8,-1]);self.assertEqual(mx,8)


class ReplayTests(unittest.TestCase):
    def runner(self,m):
        return GroupingReplay(m,geometry_mode='image-space-diagnostic',eps=50,close_distance=10)

    def test_cache_replay_matches_same_raw_live_boundary(self):
        rows=[frame(0),frame(3,poses=[None,keypoints(True)]),frame(6)]
        m=manifest();direct=self.runner(m);expected=[direct.process(copy.deepcopy(r)) for r in rows]
        with tempfile.TemporaryDirectory() as d:
            w=CacheWriter(Path(d)/'cache',m)
            for r in rows:w.append(r)
            w.close(complete=True);loaded,rr=read_cache(Path(d)/'cache')
            replay=self.runner(loaded)
            actual=[replay.process(r) for r in rr]
        self.assertEqual(expected,actual)
        self.assertEqual(actual[1]['persons'][0]['heading_status'],'stale')
        self.assertEqual(actual[1]['persons'][0]['heading_age_source_frames'],3)
        self.assertEqual(actual[0]['persons'][0]['predicted_group_id'],0)

    def test_replay_matches_original_live_ema_and_grouping_with_controlled_poses(self):
        # Exercise the retained live process_frame, without constructing its models.
        from analyze_subgroups import SubgroupAnalyzerPipeline
        live=object.__new__(SubgroupAnalyzerPipeline)
        H=np.diag([.03,.03,1.]).astype(np.float32)
        live.homography_matrix=H
        live.ALPHA_SMOOTH=.7
        live.pose_history_smooth={};live.gender_history={};live.gender_consensus={}
        live.groups_status={};live.max_group_id=-1
        rows=[frame(0),frame(3,poses=[None,keypoints(True)]),frame(6)]
        live.tracks_data={f"frame_{r['source_frame']:05d}":[{'id':p['person_id'],'bbox':p['bbox_xywh']} for p in r['persons']] for r in rows}
        live.vitpose=unittest.mock.Mock()
        live.vitpose.predict.side_effect=[[p['keypoints_xy_score'] for p in r['persons']] for r in rows]
        live.gender_predictor=unittest.mock.Mock()
        live.gender_predictor.predict.return_value=['Unknown','Unknown']
        replay=GroupingReplay(manifest(),geometry_mode='legacy-b0',eps=1.75,
                              calibration={'homography_matrix':H.tolist()})
        image=np.zeros((1080,1920,3),dtype=np.uint8)
        for row in rows:
            expected,_=live.process_frame(image,row['source_frame'])
            actual=replay.process(row)['persons']
            self.assertEqual([p['id_g'] for p in expected],[p['predicted_group_id'] for p in actual])
            for a,b in zip(expected,actual):
                np.testing.assert_array_equal(a['real_pos'],b['position_xy'])
                np.testing.assert_array_equal(live.pose_history_smooth[a['id_p']],b['smoothed_image_heading'])

    def test_empty_frame_clears_heading_and_preserves_maximum(self):
        runner=self.runner(manifest(10));runner.process(frame(0))
        runner.process(frame(3,people=[],poses=[]))
        result=runner.process(frame(6,poses=[None,None]))
        self.assertEqual(result['persons'][0]['heading_status'],'missing')
        self.assertEqual(result['persons'][0]['predicted_group_id'],1)

    def test_repeatability_exclusions_and_observation_immutability(self):
        rows=[frame(0),frame(3,people=[person(1,w=14)],poses=[]),frame(6)]
        original=copy.deepcopy(rows)
        a,b=self.runner(manifest()),self.runner(manifest())
        self.assertEqual([a.process(r) for r in rows],[b.process(r) for r in rows])
        self.assertEqual(rows,original)

    def test_replay_rejects_frame_reordering(self):
        r=self.runner(manifest());r.process(frame(0))
        with self.assertRaises(ValueError):r.process(frame(6))
        with self.assertRaises(ValueError):self.runner(manifest()).process(frame(3))

    def test_cli_replay_does_not_import_neural_modules(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d)/'cache';out=Path(d)/'replay';w=CacheWriter(cache,manifest(1));w.append(frame());w.close(complete=True)
            code='''import builtins,sys,runpy
old=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','transformers','inference','ultralytics','boxmot'):
        raise AssertionError('neural import attempted: '+name)
    return old(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=['analyze_subgroups.py','replay','--cache',sys.argv[1],'--output',sys.argv[2],'--geometry-mode','image-space-diagnostic','--eps','50','--close-distance','10']
runpy.run_path('analyze_subgroups.py',run_name='__main__')
'''
            subprocess.run([sys.executable,'-B','-c',code,str(cache),str(out)],cwd=ROOT,check=True,capture_output=True,text=True)
            completed=json.loads((out/'manifest.json').read_text())
            self.assertEqual(completed['state'],'complete')
            self.assertEqual(completed['frame_records'],1)
            self.assertEqual(completed['observation_inference_kind'],'synthetic-test')
            self.assertEqual(completed['id_namespace'],'synthetic-test')




class RealSmokeCacheTests(unittest.TestCase):
    cache=ROOT/'.local/c2/vitpose_plus_20261109/observations_verified'

    def setUp(self):
        if not self.cache.exists():self.skipTest('bounded real smoke cache not present; no inference/download in regressions')

    def test_real_smoke_frames_and_person_keypoint_association(self):
        m,rows=read_cache(self.cache)
        _,xml=read_cvat(ROOT/'WIN_20260303_10_09_56_PRO_GT_First60s/annotations.xml')
        self.assertEqual([r['source_frame'] for r in rows],[0,3,6,9,12])
        self.assertEqual(m['pose_model']['model_id'],'usyd-community/vitpose-plus-base')
        self.assertEqual(m['pose_model']['dataset_index'],0)
        self.assertEqual(len(m['pose_model']['resolved_revision']),40)
        self.assertEqual(m['pose_batch_size'],1)
        self.assertEqual(m['device']['requested'],'cuda:0')
        for row in rows:
            expected=xml[row['source_frame']]
            self.assertEqual([p['person_id'] for p in row['persons']],[p['person_id'] for p in expected])
            for p,q in zip(row['persons'],expected):
                self.assertEqual(p['bbox_xywh'],q['bbox_xywh'])
                self.assertEqual(p['reference']['attributes'],q['attributes'])
                self.assertEqual(p['pose_status'],'ok' if p['eligible'] else 'excluded')

    def test_real_cache_replay_repeatability_and_immutability(self):
        from annotation_io import file_sha256
        before={f:file_sha256(self.cache/f) for f in ('manifest.json','frames.jsonl')}
        m,rows=read_cache(self.cache)
        a=GroupingReplay(m,geometry_mode='image-space-diagnostic',eps=50,close_distance=10)
        b=GroupingReplay(m,geometry_mode='image-space-diagnostic',eps=50,close_distance=10)
        self.assertEqual([a.process(r) for r in rows],[b.process(r) for r in rows])
        self.assertEqual(before,{f:file_sha256(self.cache/f) for f in before})


class ViTPosePlusBoundaryTests(unittest.TestCase):
    def test_native_postprocess_preserves_roi_order_and_source_pixel_offsets(self):
        from types import SimpleNamespace
        import torch
        from transformers.models.vitpose.image_processing_vitpose import VitPoseImageProcessor
        processor=VitPoseImageProcessor()
        y,x=np.mgrid[:64,:48]
        gaussian=np.exp(-((x-24)**2+(y-32)**2)/8).astype(np.float32)
        maps=torch.tensor(np.tile(gaussian,(2,17,1,1)))
        poses=processor.post_process_pose_estimation(SimpleNamespace(heatmaps=maps),boxes=[[[10,10,20,40],[110,30,20,40]]])[0]
        self.assertEqual(len(poses),2)
        delta=(poses[1]['keypoints']-poses[0]['keypoints']).numpy()
        np.testing.assert_allclose(delta,np.tile([100.,20.],(17,1)),atol=1e-4)
        # This metadata is not the input pixel ROI in the installed implementation.
        self.assertFalse(np.allclose(poses[0]['bbox'].numpy(),[10,10,29,49]))

    def test_fixed_hub_model_pinned_processor_and_dataset_index_per_person(self):
        from types import SimpleNamespace
        import torch
        from transformers import BatchFeature
        from analyze_subgroups import HFViTPosePredictor,VITPOSE_MODEL
        sha='c'*40
        config=SimpleNamespace(model_type='vitpose',id2label=dict(enumerate(COCO17)),num_labels=17,
            backbone_config=SimpleNamespace(num_experts=6,part_features=192),use_simple_decoder=False)
        class Model:
            dtype=torch.float32
            def __init__(self):self.config=config;self.dataset_indices=[]
            def to(self,device):return self
            def eval(self):return self
            def __call__(self,pixel_values,dataset_index):
                self.dataset_indices.append(dataset_index.clone())
                return SimpleNamespace(heatmaps=torch.zeros(len(pixel_values),17,64,48))
        class Processor:
            def __call__(self,image,boxes,return_tensors):
                return BatchFeature({'pixel_values':torch.zeros(len(boxes[0]),3,256,192)})
            def post_process_pose_estimation(self,outputs,boxes):
                result=[]
                for x,y,w,h in boxes[0]:
                    result.append({'keypoints':torch.tensor([[x,y]]*17),'scores':torch.ones(17),
                                   'labels':torch.arange(17),'bbox':torch.tensor([x,y,x+w-1,y+h-1])})
                return [result]
        model=Model();processor=Processor()
        with tempfile.TemporaryDirectory() as d:
            artifact=Path(d)/'synthetic-artifact';artifact.write_bytes(b'SYNTHETIC TEST ONLY')
            with patch('huggingface_hub.HfApi') as api, patch('huggingface_hub.try_to_load_from_cache',return_value=str(artifact)), \
                 patch('transformers.AutoConfig.from_pretrained',return_value=config), \
                 patch('transformers.AutoProcessor.from_pretrained',return_value=processor) as load_processor, \
                 patch('transformers.VitPoseForPoseEstimation.from_pretrained',return_value=model) as load_model:
                api.return_value.model_info.return_value.sha=sha
                predictor=HFViTPosePredictor(device='cpu',strict=True)
                poses=predictor.predict(np.zeros((100,100,3),dtype=np.uint8),[[0,0,20,40],[30,0,50,40]])
                self.assertEqual(VITPOSE_MODEL,'usyd-community/vitpose-plus-base')
                self.assertEqual(load_processor.call_args.args,(VITPOSE_MODEL,))
                self.assertEqual(load_model.call_args.args,(VITPOSE_MODEL,))
                self.assertEqual(load_processor.call_args.kwargs['revision'],sha)
                self.assertEqual(load_model.call_args.kwargs['revision'],sha)
                self.assertEqual(predictor.identity['resolved_revision'],sha)
                self.assertEqual(model.dataset_indices[0].tolist(),[0,0])
                self.assertEqual(model.dataset_indices[0].dtype,torch.long)
                self.assertEqual([p.shape for p in poses],[(17,3),(17,3)])
                self.assertEqual([float(p[0,0]) for p in poses],[0,30])
                self.assertEqual(predictor.predict(np.zeros((1,1,3)),[]),[])
                with patch.object(processor,'post_process_pose_estimation') as post:
                    post.return_value=[[{'keypoints':torch.zeros(17,2),'scores':torch.ones(17),
                                         'labels':torch.arange(16,-1,-1),'bbox':torch.tensor([0,0,19,39])}]]
                    with self.assertRaises(RuntimeError):predictor.predict(np.zeros((100,100,3)),[[0,0,20,40]])


if __name__=='__main__':
    unittest.main(verbosity=2)
