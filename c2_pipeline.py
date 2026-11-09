"""Small C2 commands used by analyze_subgroups: alignment, raw cache, CPU replay."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

from annotation_io import file_sha256, probe_video, pts_seconds, read_cvat, verify_decode
from observation_cache import COCO17, SCHEMA, CacheWriter, canonical_hash, read_cache


def provenance():
    packages = {}
    for name in ('torch','transformers','numpy','opencv-python','scikit-learn'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parent
    files = ['analyze_subgroups.py','c2_pipeline.py','annotation_io.py','observation_cache.py',
             'orientation_utils.py','grouping_contracts.py','identify_group_social.py',
             'orientation_gate.py','group_identity.py']
    return {'conda_environment': 'paper-research' if Path(sys.prefix).name == 'paper-research' else 'other',
            'sys_executable': sys.executable, 'python': sys.version, 'packages': packages,
            'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'source_sha256': {f:file_sha256(root/f) for f in files}}


def write_new_json(path, value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,allow_nan=False)


def validate_data(args):
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    meta, frames = read_cvat(args.xml)
    timing = probe_video(args.video)
    if (meta['width'],meta['height']) != (timing['width'],timing['height']):
        raise ValueError('XML/video dimensions differ')
    if meta['start_frame'] != 0 or meta['stop_frame']+1 != timing['frame_count']:
        raise ValueError('XML/video frame coverage differs; explicit offset adapter required')
    timing['decoder_verification'] = verify_decode(args.video,timing,args.end,
        sample_frames=[f for f in (0,1,3,30,300,319,334,899,900,1799) if f < args.end],
        output=out/'samples',annotations=frames)
    timing['annotation_metadata'] = meta
    timing['visual_review'] = {'status':'pending', 'note':'Review sample overlays against the original video; metadata alone is insufficient.'}
    timing['provenance'] = provenance()
    write_new_json(out/'alignment.json',timing)
    print(out/'alignment.json')


def observation_frame(index, ticks, time_base, people, pose_results):
    """Raw inference boundary: no heading extraction, EMA, geometry or grouping."""
    eligible = [p for p in people if not p['outside'] and p['bbox_xywh'][2] >= 15 and p['bbox_xywh'][3] >= 30]
    if len(eligible) != len(pose_results):
        raise ValueError('pose/person count mismatch')
    poses = iter(pose_results)
    persons = []
    for person in people:
        reason = 'outside' if person['outside'] else 'tiny_bbox' if person['bbox_xywh'][2] < 15 or person['bbox_xywh'][3] < 30 else None
        kpts = next(poses) if reason is None else None
        if hasattr(kpts,'tolist'):
            kpts = kpts.tolist()
        persons.append({'person_id':person['person_id'], 'bbox_xywh':person['bbox_xywh'],
            'detection_confidence':None, 'eligible':reason is None, 'exclusion_reason':reason,
            'pose_status':'excluded' if reason else 'ok' if kpts is not None else 'missing',
            'missing_reason':reason or ('pose_unavailable' if kpts is None else None),
            'keypoints_xy_score':kpts,
            'reference':{k:person[k] for k in ('outside','occluded','keyframe','attributes')}})
    return {'source_frame':index,'pts_ticks':ticks,'timestamp_seconds':pts_seconds(ticks,time_base),
            'decode_status':'ok','persons':persons}


def export_cache(args):
    if Path(args.output).exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    if args.start != 0:
        raise ValueError('initial v1 exporter requires start=0; no implicit seek/offset')
    if args.stride < 1 or args.end <= args.start or len(range(args.start,args.end,args.stride)) > args.max_pose_frames:
        raise ValueError('invalid range/stride or bounded pose-frame limit exceeded')
    timing = json.loads(Path(args.alignment).read_text(encoding='utf8'))
    meta, frames = read_cvat(args.xml)
    if (file_sha256(args.video) != timing['video_sha256'] or meta['source_sha256'] != timing['annotation_metadata']['source_sha256']):
        raise ValueError('alignment source identity mismatch')
    if timing['decoder_verification']['verified_end_exclusive'] < args.end:
        raise ValueError('source timing has not been verified through requested range')
    if timing['visual_review']['status'] != 'sampled-alignment-supported':
        raise ValueError('sampled XML/video visual review is pending')
    if args.pose_batch_size < 1:
        raise ValueError('pose batch size must be positive')
    from analyze_subgroups import HFViTPosePredictor
    import cv2
    import torch
    import time
    if Path(sys.prefix).name != 'paper-research':
        raise RuntimeError('cache inference requires the paper-research interpreter')
    cuda = str(args.device).startswith('cuda')
    if cuda and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA is unavailable; no CPU fallback')
    if cuda:
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    loading_start = time.perf_counter()
    predictor = HFViTPosePredictor(device=args.device,strict=True,revision=args.revision)
    loading_seconds = time.perf_counter()-loading_start
    model = predictor.identity
    memory = {}
    if cuda:
        memory = {'model_load_peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device),
                  'after_model_load_allocated_bytes':torch.cuda.memory_allocated(args.device),
                  'after_model_load_reserved_bytes':torch.cuda.memory_reserved(args.device)}
        torch.cuda.reset_peak_memory_stats(args.device)
    calls = []
    inference_start = time.perf_counter()
    manifest = {'schema':SCHEMA,'joint_order':COCO17,'video_sha256':timing['video_sha256'],
        'input_sha256':meta['source_sha256'],'id_namespace':f"cvat-job-{meta['job_id']}",
        'input_kind':'oracle-cvat-boxes-and-identities','width':timing['width'],'height':timing['height'],
        'start_frame':args.start,'end_frame_exclusive':args.end,'stride':args.stride,'time_base':timing['time_base'],
        'coordinate_units':'pixels','bbox_format':'xywh','auxiliary_attributes':'disabled','inference_kind':'real',
        'pose_model':model,'provenance':provenance(), 'alignment_sha256':file_sha256(args.alignment),
        'device':{'requested':args.device,'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda},
        'pose_batch_size':args.pose_batch_size,
        'numerics':{'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark},
        'calibration':None,'eligibility':{'min_width_pixels':15,'min_height_pixels':30,'outside_excluded':True},
        'reference_policy':'partial social_group attributes retained verbatim; unreviewed is unknown'}
    cap = cv2.VideoCapture(str(args.video))
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError('cannot open source video')
        writer = CacheWriter(args.output,manifest)
        for index in range(args.end):
            ok,image = cap.read()
            if not ok:
                raise ValueError(f'decode ended at frame {index}')
            if image.shape[:2] != (manifest['height'],manifest['width']):
                raise ValueError('decoded shape mismatch')
            if abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-pts_seconds(timing['pts_ticks'][index],timing['time_base'])) > .002:
                raise ValueError('decode PTS mismatch during cache export')
            if index % args.stride:
                continue
            people = frames.get(index,[])
            eligible = [p for p in people if not p['outside'] and p['bbox_xywh'][2]>=15 and p['bbox_xywh'][3]>=30]
            boxes = [[x,y,x+w,y+h] for p in eligible for x,y,w,h in [p['bbox_xywh']]]
            rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            poses = []
            for offset in range(0,len(boxes),args.pose_batch_size):
                batch_boxes = boxes[offset:offset+args.pose_batch_size]
                predictions = predictor.predict(rgb,batch_boxes)
                if len(predictions) != len(batch_boxes):
                    raise ValueError('batch output count mismatch; refusing positional association')
                poses.extend(predictions)
                calls.append({'source_frame':index,'person_ids':[p['person_id'] for p in eligible[offset:offset+args.pose_batch_size]],
                              'input_bboxes_xywh':[p['bbox_xywh'] for p in eligible[offset:offset+args.pose_batch_size]],
                              **predictor.last_diagnostics})
            writer.append(observation_frame(index,timing['pts_ticks'][index],timing['time_base'],people,poses))
        if cuda:
            torch.cuda.synchronize(args.device)
            memory.update(inference_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                          inference_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        metrics = {'model_loading_and_download_seconds':loading_seconds,
                   'decoded_and_pose_export_seconds':time.perf_counter()-inference_start,
                   'cuda_memory':memory,'calls':calls,'warmup':'none; first-call latency retained',
                   'scope':'bounded real XML-box/ID pose inference, not benchmark accuracy'}
        write_new_json(Path(args.output)/'inference_metrics.json',metrics)
        writer.manifest['inference_metrics_sha256'] = file_sha256(Path(args.output)/'inference_metrics.json')
        writer.close(complete=True)
    finally:
        cap.release()
        if writer is not None:
            writer.close()


class GroupingReplay:
    """C1 group association with the original analyzer's heading EMA/reset policy."""
    def __init__(self, manifest, *, geometry_mode, eps, close_distance=None, calibration=None,
                 orientation_mode='B0-original', alpha=.7, confidence_threshold=.6,
                 min_samples=2, threshold_overlap=.5):
        import numpy as np
        from grouping_contracts import geometry_contract, validate_grouping_inputs
        if not 0 <= alpha <= 1 or not 0 <= confidence_threshold <= 1:
            raise ValueError('invalid orientation recipe')
        H = np.asarray(calibration['homography_matrix'],dtype=float) if calibration else None
        validate_grouping_inputs([],[],homography_matrix=H,eps=eps,min_samples=min_samples)
        self.geometry = geometry_contract(geometry_mode,H,calibration=calibration,video=manifest,close_distance=close_distance)
        self.H, self.manifest, self.calibration = H,manifest,calibration
        self.config = {'geometry_mode':geometry_mode,'eps':eps,'close_distance':self.geometry['close_distance'],
                       'orientation_mode':orientation_mode,'alpha':alpha,'confidence_threshold':confidence_threshold,
                       'min_samples':min_samples,'threshold_overlap':threshold_overlap,'state_initialization':'reset_at_cache_start'}
        self.state, self.maximum, self.history, self.last_fresh = {},-1,{},{}
        self.last_frame = None

    def process(self,row):
        import numpy as np
        from identify_group_social import cluster_bboxes_with_ids
        from grouping_contracts import project_geometry
        from orientation_utils import get_pose_vector
        from observation_cache import validate_frame
        validate_frame(row,self.manifest)
        expected_frame = self.manifest['start_frame'] if self.last_frame is None else self.last_frame+self.manifest['stride']
        if row['source_frame'] != expected_frame:
            raise ValueError('replay requires consecutive scheduled source frames')
        persons = [p for p in row['persons'] if p['eligible']]
        ids = [p['person_id'] for p in persons]
        self.history = {k:v for k,v in self.history.items() if k in ids}
        self.last_fresh = {k:v for k,v in self.last_fresh.items() if k in ids}
        headings,details = [],{}
        for p in persons:
            tid = p['person_id']
            raw = get_pose_vector(p['keypoints_xy_score'],conf_thresh=self.config['confidence_threshold'])
            if raw is not None:
                if tid in self.history:
                    a = self.config['alpha']
                    old = self.history[tid]
                    vx,vy = a*raw[0]+(1-a)*old[0],a*raw[1]+(1-a)*old[1]
                    norm = math.sqrt(vx**2+vy**2)
                    final = (vx/norm,vy/norm) if norm > 0 else raw
                else:
                    final = raw
                self.history[tid] = final
                self.last_fresh[tid] = row['source_frame']
            else:
                final = self.history.get(tid)
            headings.append(final)
            details[tid] = {'raw_image_heading':None if raw is None else list(raw),
                           'smoothed_image_heading':None if final is None else list(final),
                           'heading_status':'fresh' if raw is not None else 'stale' if final is not None else 'missing',
                           'heading_age_source_frames':None if tid not in self.last_fresh else row['source_frame']-self.last_fresh[tid]}
        boxes = [p['bbox_xywh'] for p in persons]
        c = self.config
        result,self.state,self.maximum = cluster_bboxes_with_ids(self.state,boxes,ids,self.maximum,
            pose_vectors=headings,homography_matrix=self.H,eps=c['eps'],min_samples=c['min_samples'],
            threshold_overlap=c['threshold_overlap'],orientation_mode=c['orientation_mode'],
            geometry_mode=c['geometry_mode'],calibration=self.calibration,video_identity=self.manifest,
            close_distance=c['close_distance'])
        positions,projected = project_geometry(boxes,headings,self.H,image_directions=c['geometry_mode']=='image-space-diagnostic')
        predictions = {}
        for i,p in enumerate(result):
            tid = p['id_p']
            predictions[tid] = {'predicted_group_id':p['id_g'], **details[tid],
                'position_xy':positions[i].tolist(),'coordinate_units':self.geometry['units'],
                'projected_heading':None if projected[i] is None else np.asarray(projected[i]).tolist(),
                'group_confidence':None}
        self.last_frame = row['source_frame']
        return {'source_frame':row['source_frame'],'pts_ticks':row['pts_ticks'],'timestamp_seconds':row['timestamp_seconds'],
                'persons':[{'person_id':p['person_id'],'bbox_xywh':p['bbox_xywh'],
                            **predictions.get(p['person_id'],{'predicted_group_id':None,'exclusion_reason':p['exclusion_reason']})}
                           for p in row['persons']]}


def replay_cache(args):
    manifest,rows = read_cache(args.cache)
    calibration = json.loads(Path(args.calibration).read_text(encoding='utf8')) if args.calibration else None
    runner = GroupingReplay(manifest,geometry_mode=args.geometry_mode,eps=args.eps,
        close_distance=args.close_distance,calibration=calibration,orientation_mode=args.orientation_mode)
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    run = {'schema':'grouping-replay.v1','state':'incomplete','observation_manifest_sha256':file_sha256(Path(args.cache)/'manifest.json'),
        'observations_sha256':manifest['frames_sha256'],'recipe':runner.config,'geometry':runner.geometry,
        'observation_cache':str(Path(args.cache).resolve()),'observation_inference_kind':manifest['inference_kind'],
        'video_sha256':manifest['video_sha256'],'id_namespace':manifest['id_namespace'],
        'source_range':{k:manifest[k] for k in ('start_frame','end_frame_exclusive','stride')},
        'calibration':calibration,'provenance':provenance(),'labels_are':'predictions/pre-annotations, not ground truth',
        'reference_input_kind':manifest.get('input_kind'),'source_time_base':manifest['time_base']}
    write_new_json(out/'manifest.json',run)
    with (out/'predictions.jsonl').open('x',encoding='utf8') as handle:
        for row in rows:
            handle.write(json.dumps(runner.process(row),allow_nan=False)+'\n')
    run.update(state='complete',frame_records=len(rows),predictions_sha256=file_sha256(out/'predictions.jsonl'))
    write_new_json(out/'manifest.pending',run)
    (out/'manifest.pending').replace(out/'manifest.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    align = sub.add_parser('validate-data')
    align.add_argument('--video',required=True)
    align.add_argument('--xml',required=True)
    align.add_argument('--end',type=int,default=1800)
    align.add_argument('--output',required=True)
    cache = sub.add_parser('cache')
    for name in ('video','xml','alignment','output'):
        cache.add_argument('--'+name,required=True)
    cache.add_argument('--start',type=int,default=0)
    cache.add_argument('--end',type=int,required=True)
    cache.add_argument('--stride',type=int,default=3)
    cache.add_argument('--max-pose-frames',type=int,default=5,help='Explicit safety bound; initial smoke test only')
    cache.add_argument('--device',default='cuda:0')
    cache.add_argument('--pose-batch-size',type=int,default=1)
    cache.add_argument('--revision',help='Exact HF commit SHA; by default resolve main once and pin it')
    replay = sub.add_parser('replay')
    for name in ('cache','output'):
        replay.add_argument('--'+name,required=True)
    replay.add_argument('--geometry-mode',choices=['metric','image-space-diagnostic','legacy-b0'],required=True)
    replay.add_argument('--eps',type=float,required=True,help='In explicitly selected coordinate units')
    replay.add_argument('--close-distance',type=float)
    replay.add_argument('--calibration',help='Verified calibration.v1 including homography_matrix')
    replay.add_argument('--orientation-mode',choices=['B0-original','proximity-only','signed-dyad-diagnostic'],default='B0-original')
    args = parser.parse_args(argv)
    {'validate-data':validate_data,'cache':export_cache,'replay':replay_cache}[args.command](args)


if __name__ == '__main__':
    main()
