import ast
import copy
import math
import unittest
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from modules.group_identity import assign_group_ids
from modules.orientation_gate import compatible
from diagnostics.core import evaluate_detection, match, require_dense, validate
from diagnostics.cli import filter_cache, mot_export, oracle, prepare_trackeval
from diagnostics.timing import select_interval


def doc(kind, items, start=0):
    return {'schema_version': 1, 'kind': kind, 'annotation_status': 'human_reviewed' if kind=='annotations' else None,
            'source': {'video': 'synthetic', 'start_frame': start, 'end_frame_exclusive': start+1},
            'frames': [{'frame': start, 'coverage': 'full', 'items': items}]}


def person(pid=1, bbox=None, **extra):
    return {'person_id': pid, 'bbox': bbox or [0,0,10,10], **extra}


def execute_id_block(source, old, clusters):
    """Execute actual archived/current assignment statements without model imports."""
    tree = ast.parse(source)
    f = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='cluster_bboxes_with_ids')
    def assigns(n, name):
        return isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name
                                               for target in n.targets for x in ast.walk(target))
    start = next(i for i,n in enumerate(f.body) if assigns(n,'new_group_status'))
    end = next(i for i,n in enumerate(f.body) if assigns(n,'n_clusters'))
    scope = {'groups_status': old, 'cluster_members': clusters, 'threshold_overlap': 0.5,
             'max_group_id': max(old, default=-1), 'defaultdict': defaultdict,
             'assign_group_ids': assign_group_ids}
    exec(compile(ast.Module(body=f.body[start:end], type_ignores=[]),'<assignment>','exec'),scope)
    return scope['new_group_status']


class CorrectnessTests(unittest.TestCase):
    def test_variable_rate_frame_selection_uses_timestamps_not_nominal_fps(self):
        ts={'frames':[{'frame':n,'timestamp_seconds':t} for n,t in enumerate([0,0.03,0.08,0.19,0.22])]}
        self.assertEqual(select_interval(ts,0.05,0.2),(2,4,0.08,0.22))
        self.assertRaises(ValueError,select_interval,ts,0.2,0.3)
    def test_archived_split_overwrites_both_modules_and_fix_preserves_members(self):
        for name in ('identify_group.py','identify_group_social.py'):
            old = {7: [1,2,3,4]}; current = {0:[1,2], 1:[3,4]}
            broken = execute_id_block(Path('diagnostics/baselines',name).read_text(encoding='utf8'),old,current)
            fixed = execute_id_block(Path('modules',name).read_text(encoding='utf8'),old,current)
            self.assertEqual(len(broken),1)
            self.assertEqual(len(fixed),2)
            self.assertEqual(sorted(p for v in fixed.values() for p in v),[1,2,3,4])
            self.assertEqual(fixed[7],[1,2])

    def test_merge_birth_ties_and_monotonic_ids(self):
        groups, counter, events = assign_group_ids({2:[1,2],8:[3,4]}, {0:[1,2,3,4],1:[5,6]},8)
        self.assertEqual(groups,{2:[1,2,3,4],9:[5,6]})
        self.assertEqual(counter,9); self.assertTrue(events[0]['merge'])
        self.assertEqual(assign_group_ids({}, {},counter)[1],counter)
        self.assertRaises(ValueError,assign_group_ids,{}, {0:[1,2],1:[2,3]},-1)

    def test_legacy_gate_nonselective_grid_and_signed_candidate(self):
        for a in range(0,360,5):
            for b in range(0,360,5):
                pi=(math.cos(math.radians(a)),math.sin(math.radians(a)))
                pj=(math.cos(math.radians(b)),math.sin(math.radians(b)))
                self.assertTrue(compatible(pi,pj,(1,0),'legacy'))
        self.assertTrue(compatible((1,0),(-1,0),(1,0),'facing_experimental',1,1))
        self.assertFalse(compatible((-1,0),(1,0),(1,0),'facing_experimental',1,1))
        self.assertTrue(compatible((-1,0),(1,0),(1,0),'facing_experimental',None,None))

    def test_matching_finds_maximum_cardinality_not_greedy(self):
        gt=[person(1,[0,0,10,10]),person(2,[5,0,10,10])]
        pred=[person(3,[2,0,10,10]),person(4,[0,0,10,10])]
        self.assertEqual(len(match(pred,gt,0.5)),2)

    def test_detection_duplicates_misses_and_ignored_regions(self):
        gt=doc('annotations',[person(),person(2,[30,0,10,10]),person(None,[60,0,10,10],ignore=True)])
        pred=doc('detections',[person(),person(),person(3,[90,0,10,10]),person(4,[60,0,10,10])])
        r=evaluate_detection(pred,gt)
        self.assertEqual([r[k] for k in ('tp','fp','fn','duplicates','ignored_predictions')],[1,2,1,1,1])
        self.assertEqual(r['precision'],1/3);self.assertEqual(r['recall'],0.5)

    def test_unknown_frame_or_source_never_silently_scored(self):
        p=doc('detections',[]);g=doc('annotations',[])
        p['frames']=[]
        self.assertRaises(ValueError,evaluate_detection,p,g)
        p=doc('detections',[]);p['source']['video']='other'
        self.assertRaises(ValueError,evaluate_detection,p,g)
        g['frames'][0]['coverage']='pending'
        self.assertRaises(ValueError,evaluate_detection,doc('detections',[]),g)

    def test_oracle_preserves_misses_drops_unmatched_and_gt_boxes_are_detections(self):
        g=doc('annotations',[person(),person(2,[30,0,10,10])])
        p=doc('detections',[person(99,confidence=0.8),person(98,confidence=0.7)])
        r=oracle(g,'matched_detection_ids',p)
        self.assertEqual(len(r['frames'][0]['items']),1)
        self.assertEqual(r['frames'][0]['items'][0]['person_id'],1)
        self.assertEqual(r['oracle']['exclusions'],{'dropped_unmatched_detections':1,'missed_gt_people':1})
        self.assertEqual(oracle(g,'gt_boxes')['kind'],'detections')
        self.assertNotIn('person_id',oracle(g,'gt_boxes')['frames'][0]['items'][0])

    def test_sparse_or_unresolved_annotations_cannot_export_tracking(self):
        g=doc('annotations',[person(None)])
        self.assertRaises(ValueError,require_dense,g,True)
        g=doc('annotations',[person()]);g['source']['end_frame_exclusive']=2
        self.assertRaises(ValueError,require_dense,g,True)
        g=doc('annotations',[person(identity_uncertain=True)])
        self.assertRaises(ValueError,require_dense,g,True)

    def test_trackeval_preparation_schema_and_human_review_gate(self):
        g=doc('annotations',[person()]);g['source'].update(fps=30,width=1920,height=1080)
        t=doc('tracks',[person(9,confidence=0.8)]);t['source']=copy.deepcopy(g['source'])
        with TemporaryDirectory() as d:
            root=Path(d)/'evaluation'
            prepare_trackeval(g,t,root,'pilot01')
            self.assertEqual(len((root/'gt/pilot01/gt/gt.txt').read_text().strip().split(',')),9)
            self.assertEqual(len((root/'trackers/estimated/data/pilot01.txt').read_text().strip().split(',')),10)
            self.assertIn('seqLength=1',(root/'gt/pilot01/seqinfo.ini').read_text())
        g['annotation_status']='pending'
        self.assertRaises(ValueError,require_dense,g,True)

    def test_mot_frame_and_id_conversion_empty_frame(self):
        g=doc('annotations',[person(0)],start=100)
        g['source']['end_frame_exclusive']=102
        g['frames'].append({'frame':101,'items':[],'coverage':'full'})
        with TemporaryDirectory() as d:
            p=Path(d)/'gt.txt';mot_export(g,p)
            self.assertEqual(p.read_text().split(',')[:2],['1','1'])
            self.assertEqual(p.read_text().strip().split(',')[-1],'-1')
            self.assertTrue(Path(str(p)+'.metadata.json').exists())

    def test_sweep_does_not_invent_discarded_detections(self):
        d=doc('detections',[person(confidence=0.1),person(2,confidence=0.3)])
        d['cache']={'confidence_floor':0.1}
        self.assertEqual(len(filter_cache(d,0.2)['frames'][0]['items']),1)
        self.assertRaises(ValueError,filter_cache,d,0.05)

    def test_duplicate_identity_and_invalid_box_rejected(self):
        self.assertRaises(ValueError,validate,doc('tracks',[person(),person()]))
        self.assertRaises(ValueError,validate,doc('detections',[person(bbox=[0,0,-1,10])]))


if __name__ == '__main__':
    unittest.main()
