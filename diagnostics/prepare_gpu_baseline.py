"""Materialize a portable, bounded baseline config on the GPU server; no inference."""
import argparse
import copy
from pathlib import Path
from .core import digest, freeze, read, require_dense, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',required=True);p.add_argument('--timestamps',required=True)
    p.add_argument('--video',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--tracker-config',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();out=Path(a.out_dir)
    if out.exists():raise FileExistsError('Choose a new baseline folder')
    gt=read(a.annotations);require_dense(gt,annotated=True)
    timing=read(a.timestamps)
    for path in (a.video,a.checkpoint,a.tracker_config):
        if not Path(path).is_file():raise FileNotFoundError(path)
    sha=digest(a.video)
    if sha!=gt['source']['video_sha256'] or sha!=timing['video_sha256']:
        raise ValueError('Server source video differs from annotated source')
    start,end=gt['source']['start_frame'],gt['source']['end_frame_exclusive']
    duration=timing['frames'][end]['timestamp_seconds']-timing['frames'][start]['timestamp_seconds']
    if not 0<duration<=40:raise ValueError('This diagnostic baseline is limited to 40 seconds')
    video=str(Path(a.video).resolve())
    gt=copy.deepcopy(gt);gt['source']['video']=video
    gt['portability']={'original_annotation_sha256':digest(a.annotations),
                       'change':'source path relocated only; source video SHA256 preserved'}
    timing=copy.deepcopy(timing);timing['video']=video
    config={'status':'development diagnostic baseline; no historical-run equivalence claim',
      'pilot':{'video':video,'start_frame':start,'end_frame_exclusive':end,
               'max_duration_seconds':40,'frame_timestamp_file':str((out/'timestamps.json').resolve()),
               'scene':gt['source'].get('scene'),'session':gt['source'].get('session')},
      'detector':{'checkpoint':str(Path(a.checkpoint).resolve()),'device':'cuda:0','half':False,
                  'imgsz':1280,'confidence_floor':0.15,'nms_iou':0.45,'max_det':300},
      'tracker':{'type':'bytetrack','config_file':str(Path(a.tracker_config).resolve()),
                 'device':'cuda:0','half':True,'input_confidence':0.15},
      'scope':'detector + tracker only, fresh tracker at first scored frame; no stitching/pose/grouping',
      'legacy_baseline':{'scripts':['detect_yolo.py','track.py','stitch_tracks.py'],
                         'tracker_post_construction_overrides':{'track_thresh':0.25,'track_buffer':90},
                         'tracker_input_gate':'confidence > 0.15',
                         'bbox_rounding':'original integer rounding preserved',
                         'stitching':{'MAX_TIME_LOST':150,'MAX_SPATIAL_DIST':350.0,'MIN_SIMILARITY':0.60}},
      'operating_point_basis':'matches audited offline detection script settings; actual weights/YAML supplied explicitly',
      'ground_truth_sha256':digest(a.annotations)}
    lock=freeze(config,[a.annotations,a.timestamps,a.video,a.checkpoint,a.tracker_config])
    write(out/'config.json',config);write(out/'annotations.json',gt)
    write(out/'timestamps.json',timing);write(out/'config_lock.json',lock)
    write(out/'source.json',gt['source'])
    from .legacy_baseline import generate
    generate(config,out)
    print('Prepared without inference:',out)


if __name__=='__main__':main()
