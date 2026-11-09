"""C1 regressions. No model initialization, data downloads or historical writes."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

from group_identity import associate_groups
from orientation_gate import pair_compatibility, original_absolute_predicate
from grouping_contracts import validate_grouping_inputs
from report_metrics import SourceTimebase, convex_hull_area, build_report
from generate_report import main as report_main, write_report
from orientation_utils import get_pose_vector, _post_process_vector

ROOT=Path(__file__).resolve().parent.parent
EVIDENCE={}


def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list,set)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    return v


class Case(unittest.TestCase):
    mode='native production helpers; no substitute'
    def check(self,inputs,expected,actual):
        EVIDENCE.setdefault(self.id(),[]).append({'input':clean(inputs),'expected':clean(expected),
            'actual':clean(actual),'dependency_mode':self.mode})
        self.assertEqual(clean(actual),clean(expected))
    def raises(self,fn,inputs,exception=ValueError):
        try:
            fn()
        except exception as exc:
            self.check(inputs,exception.__name__,type(exc).__name__)
            EVIDENCE[self.id()][-1]['message']=str(exc)
        else:
            self.fail(f'expected {exception.__name__}')


class GroupState(Case):
    def test_S1_continuation(self):
        value=associate_groups({7:[1,2,3]},{99:[1,2,3]},7)
        self.check('old 7:[1,2,3], raw99:[1,2,3]',({99:7},{7:[1,2,3]},7),value)
    def test_S2_split(self):
        value=associate_groups({7:[1,2,3,4]},{9:[3,4],2:[1,2]},7)
        self.check('split old7 into [1,2] / [3,4]',({2:7,9:8},{7:[1,2],8:[3,4]},8),value)
    def test_S3_merge(self):
        self.check('merge old7:[1,2],8:[3,4]',({5:7},{7:[1,2,3,4]},8),
            associate_groups({8:[3,4],7:[1,2]},{5:[1,2,3,4]},8))
    def test_S4_unrelated(self):
        self.check('max8 old[1,2] new[10,11]/[12,13]',({0:9,1:10},{9:[10,11],10:[12,13]},10),
            associate_groups({7:[1,2]},{0:[10,11],1:[12,13]},8))
    def test_S5_global_ties(self):
        value=associate_groups({8:[3,4],7:[1,2]},{5:[2,4],4:[1,3]},8)
        self.check('equal intersections/coverages',({4:7,5:8},{7:[1,3],8:[2,4]},8),value)
    def test_S6_permutation(self):
        a=associate_groups({7:[1,2,3,4]},{0:[1,2],1:[3,4]},7)[1:]
        b=associate_groups({7:[4,3,2,1]},{70:[4,3],99:[2,1]},7)[1:]
        self.check('raw labels/order/member order permuted',a,b)
    def test_E1_empty_previous(self):
        self.check('empty state max7 + new pair',({42:8},{8:[1,2]},8),associate_groups({},{42:[1,2]},7))
    def test_E2_empty_intervals(self):
        state,maximum={7:[1,2]},7
        for _ in range(4):
            _,state,maximum=associate_groups(state,{},maximum)
        self.check('four empty frames then unrelated pair',({0:8},{8:[9,10]},8),associate_groups(state,{0:[9,10]},maximum))
    def test_E3_all_noise(self):
        self.check('no raw non-noise groups, max7',({}, {},7),associate_groups({7:[1,2]},{},7))
    def test_E4_disappearance(self):
        self.check('raw0 unrelated to prior7',8,associate_groups({7:[1,2]},{0:[3,4]},7)[0][0])
    def test_E5_new_run(self):
        self.check('explicit new namespace {}, max=-1',({9:0},{0:[5,6]},0),associate_groups({},{9:[5,6]},-1))
    def test_E6_split_empty_new(self):
        _,state,m=associate_groups({7:[1,2,3,4]},{0:[1,2],1:[3,4]},7)
        _,state,m=associate_groups(state,{},m)
        self.check('split then gap then two new pairs',[9,10],sorted(associate_groups(state,{8:[20,21],7:[22,23]},m)[0].values()))
    def test_asymmetric_coverage_not_iou(self):
        self.check('old dyad subset of large new group',7,associate_groups({7:[1,2]},{0:[1,2,3,4,5,6]},7)[0][0])
    def test_ineligible_edges_and_max(self):
        self.check('insufficient old coverage and max above active ids',101,associate_groups({7:[1,2,3,4]},{0:[1,5]},100)[0][0])
    def test_invalid_partition(self):
        self.raises(lambda:associate_groups({},{0:[1,2],1:[2,3]},7),'person in two current clusters')
    def test_invalid_policy(self):
        self.raises(lambda:associate_groups({},{},7,matching_policy='iou'),'unsupported IoU policy')


class Orientation(Case):
    def test_controls_and_configured_modes(self):
        cases={'face_to_face':([1,0],[-1,0],True),'back_to_back':([-1,0],[1,0],False),
            'same_facing_away':([-1,0],[-1,0],False),'parallel':([1,0],[1,0],False),
            'side_by_side':([0,1],[0,1],False),'perpendicular':([1,0],[0,1],False)}
        for name,(a,b,signed) in cases.items():
            observed=[pair_compatibility(a,b,[1,0],1.75,m)['compatible'] for m in ('B0-original','proximity-only','signed-dyad-diagnostic')]
            self.check(name,[True,True,signed],observed)
    def test_analytical_boundaries(self):
        a=np.arccos(.3);p=[np.cos(a),np.sin(a)];q=[np.cos(np.pi-a),np.sin(np.pi-a)]
        r=pair_compatibility(p,q,[1,0],1.75)
        self.check('both directional abs near .3; mutual >=.82',True,r['compatible'])
        self.assertGreaterEqual(abs(r['cosines'][2]),.82-1e-14)
    def test_original_thresholds_strict(self):
        self.check('exact cosine thresholds (not unit-planar-consistent triple)',False,bool(original_absolute_predicate(.3,.3,.7)))
    def test_angular_sweep(self):
        a=np.deg2rad(np.arange(3600)/10);false=0
        for block in np.array_split(a,36):
            false+=int((~original_absolute_predicate(np.cos(block[:,None]),-np.cos(a[None,:]),np.cos(block[:,None]-a[None,:]))).sum())
        self.check('12,960,000 unit planar angular pairs; .1deg step',0,false)
    def test_random_sweep(self):
        a,b=np.random.default_rng(20261109).uniform(-np.pi,np.pi,(2,1000000))
        self.check('1,000,000 pairs seed20261109',0,int((~original_absolute_predicate(np.cos(a),-np.cos(b),np.cos(a-b))).sum()))
    def test_missing_and_zero(self):
        for value in (None,[0,0]):
            self.check({'heading':value},[True,False],[pair_compatibility(value,[1,0],[1,0],1.75,m)['compatible'] for m in ('B0-original','signed-dyad-diagnostic')])
    def test_invalid_directions(self):
        for v in ([float('nan'),0],[float('inf'),0],[1],[1,2,3],[[1,0]]):
            self.raises(lambda v=v:pair_compatibility(v,[1,0],[.1,0],1.75),{'invalid':repr(v),'close_bypass_must_not_hide_invalid':True})
    def test_distance_boundaries(self):
        for distance,expected in ((.49,True),(.5,False),(1.75,False),(1.750001,False)):
            self.check({'back_to_back_distance':distance},expected,pair_compatibility([-1,0],[1,0],[distance,0],1.75,'signed-dyad-diagnostic')['compatible'])
        self.check('face-to-face exactly eps',True,pair_compatibility([1,0],[-1,0],[1.75,0],1.75)['compatible'])
    def test_keypoints_validation(self):
        self.check('missing and empty keypoints',[None,None],[get_pose_vector(None),get_pose_vector([])])
        for value in (np.zeros((2,3)),np.zeros((17,2)),np.full((17,3),np.nan)):
            self.raises(lambda v=value:get_pose_vector(v),{'keypoints_shape':value.shape})
    def test_postprocess_zero_and_nonmutation(self):
        v=np.array([1.,1.]);old=v.copy();result=_post_process_vector(v)
        self.check('postprocess does not mutate source',old.tolist(),v.tolist())
        self.check('zero postprocess',None,_post_process_vector([0,0]))
        self.assertAlmostEqual(float(np.linalg.norm(result)),1.)
        self.raises(lambda:_post_process_vector([float('inf'),0]),'nonfinite postprocess vector')


def person(pid,gid=7,pos=None,gender='Unknown'):
    return {'id_p':pid,'id_g':gid,'real_pos':pos,'gender':gender,'bbox':[0,0,10,30]}


def report(data,fps=30,stride=1,**kwargs):
    return build_report(data,SourceTimebase(fps),coordinate_units='m',expected_stride=stride,**kwargs)


class Reports(Case):
    def test_source_fps_variants(self):
        for fps in (30,27.848645,7):
            r=report({'frame_00000':[person(1)],'frame_00003':[person(1)]},fps,3)
            self.check({'fps':fps,'source_frames':[0,3],'stride':3},3/fps,r['group_sessions'][0]['observed_span_seconds'])
    def test_timestamps_priority_nonuniform(self):
        tb=SourceTimebase(30,{0:100.,3:100.125,6:100.425},timestamp_origin=100.)
        r=build_report({f'frame_{f}':[person(1)] for f in (0,3,6)},tb,coordinate_units='m',expected_stride=3)
        self.check('timestamps override nominal fps',.425,round(r['group_sessions'][0]['observed_span_seconds'],10))
    def test_missing_timestamp_no_fallback(self):
        self.raises(lambda:SourceTimebase(30,{0:0.}).seconds(3),'timestamp absent even though fps supplied')
    def test_explicit_timebase_required(self):
        self.raises(lambda:SourceTimebase(),'no default fps')
        self.raises(lambda:SourceTimebase(timestamps={0:1,1:1}),'nonincreasing timestamps')
    def test_missing_frames_no_duration_bridge(self):
        r=report({'frame_0':[person(1)],'frame_3':[person(1)],'frame_12':[person(1)]},30,3)
        self.check('missing expected source frames6,9',[.1,0.],[s['observed_span_seconds'] for s in r['group_sessions']])
    def test_explicit_empty_frame_breaks_session(self):
        r=report({'frame_0':[person(1)],'frame_1':[],'frame_2':[person(1)]})
        self.check('empty source frame between same numeric ID',[0.,0.],[s['observed_span_seconds'] for s in r['group_sessions']])
    def test_time_gap_breaks_nominal_adjacency(self):
        r=build_report({'frame_0':[person(1)],'frame_1':[person(1)]},SourceTimebase(timestamps={0:0,1:5}),coordinate_units='m',expected_stride=1)
        self.check('adjacent source IDs but 5sec missing time',2,len(r['group_sessions']))
    def test_single_sample_zero_duration(self):
        self.check('single frame does not imply .1 sec',0.,report({'frame_42':[person(1)]})['group_sessions'][0]['observed_span_seconds'])
    def test_split_merge_not_formation_events(self):
        data={'frame_0':[person(i,7) for i in range(1,5)],
              'frame_1':[person(1,7),person(2,7),person(3,8),person(4,8)],
              'frame_2':[person(i,7) for i in range(1,5)]}
        r=report(data)
        self.check('split/merge snapshots; no event annotations',(2,4,None),
            (r['interval_metrics'][0]['distinct_persistent_group_ids'],len(r['group_sessions']),r['interval_metrics'][0]['group_formation_events']))
        self.check('no support across membership changes',0.,r['interval_metrics'][0]['observed_group_person_seconds'])
    def test_area_known_and_degenerate(self):
        for pts,area in (([[0,0],[1,0]],None),([[0,0],[2,0],[0,2]],2.),([[0,0],[1,0],[2,0]],0.)):
            self.check({'positions':pts},area,convex_hull_area(pts))
    def test_moving_group_not_pooled_area(self):
        a=[[0,0],[2,0],[0,2]];b=[[x+100,y] for x,y in a]
        data={'frame_0':[person(i,7,p) for i,p in enumerate(a)],'frame_1':[person(i,7,p) for i,p in enumerate(b)]}
        r=report(data)
        self.check('triangle translates100 units',[2.,2.],[g['area'] for g in r['group_observations']])
        self.check('time average / max / person time',(2.,2.,.1),
            (r['interval_group_metrics'][0]['temporal_average_area'],r['interval_group_metrics'][0]['maximum_observed_area'],r['interval_metrics'][0]['observed_group_person_seconds']))
    def test_unknowns_and_missing_geometry_retained(self):
        data={'frame_0':[person(1,7,[0,0],'Male'),person(2,7,[1,0]),person(3,7),person(4,-1),person(5,None)]}
        r=report(data)
        self.check('Unknown attribute does not remove members',(5,3,2,None,1,1),
            (len(r['person_observations']),r['group_observations'][0]['cardinality'],r['group_observations'][0]['unknown_attribute_count'],r['group_observations'][0]['area'],r['frame_metrics'][0]['ungrouped_person_count'],r['frame_metrics'][0]['unknown_membership_count']))
    def test_nonuniform_area_time_weighting(self):
        data={}
        for f,area in ((0,1),(1,3),(2,9)):
            data[f'frame_{f}']=[person(0,7,[0,0]),person(1,7,[2,0]),person(2,7,[0,area])]
        r=build_report(data,SourceTimebase(timestamps={0:0,1:.1,2:.4}),coordinate_units='m',expected_stride=1)
        self.check('area1*.1+area3*.3 over .4',2.5,round(r['interval_group_metrics'][0]['temporal_average_area'],12))
    def test_interval_half_open_and_clipping(self):
        data={'frame_0':[person(0)],'frame_1':[person(0)]}
        r=build_report(data,SourceTimebase(timestamps={0:.9,1:1.1}),coordinate_units='m',expected_stride=1,interval_seconds=1)
        self.check('support .9->1.1 clipped at1sec',[.1,.1],[round(x['observed_group_person_seconds'],8) for x in r['interval_metrics']])
        r=build_report({'frame_1':[person(0)]},SourceTimebase(1),coordinate_units='m',expected_stride=1,interval_seconds=1)
        self.check('t=1 belongs [1,2)',1,r['interval_metrics'][0]['interval_index'])
    def test_empty_report(self):
        r=report({})
        self.check('empty JSON',{k:[] for k in r if k!='metadata'},{k:v for k,v in r.items() if k!='metadata'})
    def test_json_cli_and_nonoverwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.local/c1_validation', prefix='c1-test-') as d:
            inp,out=Path(d)/'input.json',Path(d)/'report.json';inp.write_text('{}')
            report_main(['--input',str(inp),'--output',str(out),'--fps','7','--stride','3','--coordinate-units','m'])
            self.check('synthetic empty CLI report','C1-report-v1',json.loads(out.read_text())['metadata']['schema'])
            self.raises(lambda:write_report({},out),'existing report path',FileExistsError)
    @unittest.skipUnless(importlib.util.find_spec('pandas') and importlib.util.find_spec('openpyxl'),'native pandas/openpyxl unavailable; no substitute writer')
    def test_native_excel_export(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.local/c1_validation', prefix='c1-test-') as d:
            path=Path(d)/'C1.xlsx';write_report(report({}),path)
            self.check('empty native Excel export',True,path.is_file())


class Contracts(Case):
    def test_lists_and_numpy(self):
        a=validate_grouping_inputs([[0,0,10,20]],[1])[:2]
        b=validate_grouping_inputs(np.array([[0,0,10,20]]),np.array([1]))[:2]
        self.check('list vs ndarray bbox/ID',a,b)
        self.check('empty ndarray inputs',( [],[],None),validate_grouping_inputs(np.empty((0,4)),np.array([],dtype=int)))
    def test_bad_bboxes_and_ids(self):
        for boxes,ids in (([[0,0,1]],[1]),([[0,0,0,1]],[1]),([[0,np.nan,1,1]],[1]),([[0,0,1,1]],[]),([[0,0,1,1],[1,1,1,1]],[1,1])):
            self.raises(lambda b=boxes,i=ids:validate_grouping_inputs(b,i),repr((boxes,ids)))
    def test_bad_calibration_and_lengths(self):
        for kwargs in ({'pose_vectors':[]},{'pose_vectors':42},{'homography_matrix':np.zeros((3,3))},{'homography_matrix':np.eye(2)},{'scale_factor':0}):
            self.raises(lambda kw=kwargs:validate_grouping_inputs([[0,0,1,1]],[1],**kw),repr(kwargs))


class GraphDouble:
    """Substitute only: min_samples=2 neighborhood graph, not sklearn evidence."""
    matrices=[]
    def __init__(self,eps,min_samples,metric='euclidean'):
        if min_samples!=2: raise AssertionError('substitute only supports min_samples=2')
        self.eps,self.metric=eps,metric
    def fit_predict(self,x):
        d=x if self.metric=='precomputed' else np.linalg.norm(x[:,None]-x[None,:],axis=-1)
        type(self).matrices.append(np.asarray(d).copy())
        edge=d<=self.eps;labels=np.full(len(d),-1,dtype=int);gid=0
        for i in range(len(d)):
            if labels[i]>=0 or edge[i].sum()<2: continue
            labels[i]=gid;queue=[i]
            while queue:
                for j in np.where(edge[queue.pop()])[0]:
                    if labels[j]<0: labels[j]=gid;queue.append(j)
            gid+=1
        return labels


@contextmanager
def grouping_module(name,native=False,baseline=False):
    if name != 'identify_group_social':
        raise ValueError('C1 tests load only the authoritative active grouper')
    path=ROOT/(name+'.py')
    if baseline:
        manifest=json.loads((ROOT/'docs/repository_state.json').read_text(encoding='utf8'))
        source=subprocess.check_output(['git','show',manifest['original_master']+':'+name+'.py'],cwd=ROOT)
        module=types.ModuleType('_c1_baseline_'+name)
        def load():
            exec(compile(source, 'B0-original/'+name+'.py', 'exec'),module.__dict__)
    else:
        spec=importlib.util.spec_from_file_location('_c1_'+name,path)
        module=importlib.util.module_from_spec(spec)
        def load():
            spec.loader.exec_module(module)
    if native:
        load();yield module
    else:
        parent=types.ModuleType('sklearn');child=types.ModuleType('sklearn.cluster');child.DBSCAN=GraphDouble
        with patch.dict('sys.modules',{'sklearn':parent,'sklearn.cluster':child}):
            load();yield module


class GroupingProduction(Case):
    mode='production module + native NumPy/OpenCV; explicitly substituted GraphDouble DBSCAN'
    def test_S7_noise_retained_and_empty_lifecycle(self):
        for name in ('identify_group_social',):
            with grouping_module(name) as m:
                r,state,mx=m.cluster_bboxes_with_ids({7:[1,2]},np.array([[0,0,1,1]]),np.array([1]),7)
                self.check(name+' singleton',(1,-1,{},7),(len(r),r[0]['id_g'],state,mx))
                self.check(name+' empty',([],{},7),m.cluster_bboxes_with_ids({7:[1,2]},[],[],7))
    def test_split_real_entrypoints(self):
        boxes=[[0,0,.1,.1],[1,0,.1,.1],[10,0,.1,.1],[11,0,.1,.1]]
        for name in ('identify_group_social',):
            with grouping_module(name) as m:
                r,state,mx=m.cluster_bboxes_with_ids({7:[1,2,3,4]},boxes,[1,2,3,4],7,eps=1.75)
                self.check(name+' split',[7,7,8,8],[p['id_g'] for p in r])
    def test_full_modes_invoke_predicates(self):
        with grouping_module('identify_group_social') as m:
            for mode,expected in (('B0-original',True),('proximity-only',True),('signed-dyad-diagnostic',False)):
                with patch.object(m,'pair_compatibility',wraps=m.pair_compatibility) as spy:
                    r,_,_=m.cluster_bboxes_with_ids({},[[0,0,.1,.1],[1,0,.1,.1]],[1,2],-1,pose_vectors=[[-1,0],[1,0]],homography_matrix=np.eye(3),orientation_mode=mode)
                    self.check(mode+' dispatch',mode,spy.call_args.kwargs['mode'])
                    self.check(mode+' partition',expected,r[0]['id_g']>=0)
    def test_b0_valid_matrix_equivalence(self):
        for name in ('identify_group_social',):
            args=({},[[0,0,.1,.1],[1,0,.1,.1],[2,0,.1,.1],[8,0,.1,.1]],[1,2,3,4],-1)
            with grouping_module(name,baseline=True) as b:
                b.cluster_bboxes_with_ids(*args,eps=1.75)
                before=GraphDouble.matrices[-1].copy()
            with grouping_module(name) as m:
                m.cluster_bboxes_with_ids(*args,eps=1.75)
                self.check(name+' valid B0 matrix',before.tolist(),GraphDouble.matrices[-1].tolist())
    def test_invalid_inputs_never_become_edges(self):
        with grouping_module('identify_group_social') as m:
            for kwargs in ({'pose_vectors':[[np.nan,0],[1,0]]},{'pose_vectors':[[1],[0,1]]},{'pose_vectors':[]},{'eps':1000}):
                self.raises(lambda kw=kwargs:m.cluster_bboxes_with_ids({},[[0,0,1,1],[1,0,1,1]],[1,2],-1,**kw),repr(kwargs))
    def test_horizon_rejected(self):
        with grouping_module('identify_group_social') as m:
            H=np.array([[1.,0,0],[0,1,0],[0,1,-1]])
            self.raises(lambda:m.cluster_bboxes_with_ids({},[[0,0,1,1]],[1],-1,homography_matrix=H),'bbox footpoint on projective horizon')


@unittest.skipUnless(importlib.util.find_spec('sklearn'),'native sklearn unavailable; graph substitute is not native evidence')
class NativeGrouping(Case):
    mode='native sklearn + NumPy/OpenCV production module; no substitutes'
    def test_native_chain(self):
        with grouping_module('identify_group_social',native=True) as m:
            r,_,_=m.cluster_bboxes_with_ids({},[[x,0,.1,.1] for x in (0,1,2)],[1,2,3],-1,eps=1.75)
            self.check('A-B/B-C within radius, A-C outside',[0,0,0],[p['id_g'] for p in r])
    def test_native_split_and_noise(self):
        with grouping_module('identify_group_social',native=True) as m:
            r,state,mx=m.cluster_bboxes_with_ids({7:[1,2,3,4]},[[x,0,.1,.1] for x in (0,1,10,11,50)],[1,2,3,4,5],7,eps=1.75)
            self.check('native split plus noise',[7,7,8,8,-1],[p['id_g'] for p in r])


class Preservation(Case):
    def test_b0_hashes(self):
        manifest=json.loads((ROOT/'docs/repository_state.json').read_text(encoding='utf8'))
        expected=manifest['baseline_source_sha256']
        actual={f:hashlib.sha256(subprocess.check_output(['git','show',manifest['original_master']+':'+f],cwd=ROOT)).hexdigest() for f in expected}
        self.check('all nine active B0 Git blob SHA256 hashes',expected,actual)
    def test_active_helper_imports_and_calls(self):
        expected={'identify_group_social.py':{'group_identity':'associate_groups',
            'orientation_gate':'pair_compatibility','grouping_contracts':'validate_grouping_inputs'},
            'generate_report.py':{'report_metrics':'build_report'}}
        for file,helpers in expected.items():
            tree=ast.parse((ROOT/file).read_text(encoding='utf8'))
            imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
            calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
            for helper,call in helpers.items():
                self.check({'active_source':file,'helper':helper,'function':call},[True,True],
                    [helper in imports,call in calls])
    def test_no_legacy_changes_in_active_patch(self):
        baseline='master'
        excluded=['identify_group.py','tracking_group.py','run_mixste_offline.py',
            'tracking_yolov12.py','run_all_detectron2.py','run_all_yolov12.py',
            'yolo_detection_poseformer.py','test_combine_with_pose_estimation.py']
        diff=subprocess.check_output(['git','diff',baseline,'--name-only','--',*excluded],cwd=ROOT,text=True)
        self.check('excluded source files relative to clean master',[],diff.splitlines())
        self.check('retired source absent from working tree',[],[p for p in excluded+['mixste_frames.py'] if (ROOT/p).exists()])
