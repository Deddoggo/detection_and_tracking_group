"""Read-only full C2 artifact acceptance; writes only a new requested evidence file."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from annotation_io import file_sha256, read_cvat
from observation_cache import read_cache


def validate_full(cache, video, xml, alignment, execution):
    cache = Path(cache)
    m,rows = read_cache(cache)
    meta,reference = read_cvat(xml)
    timing = json.loads(Path(alignment).read_text(encoding='utf8'))
    run = json.loads(Path(execution).read_text(encoding='utf8'))
    assert run['exit_code'] == 0, 'export process failed'
    assert [r['source_frame'] for r in rows] == list(range(0,1800,3)), 'scheduled frames differ'
    assert (m['start_frame'],m['end_frame_exclusive'],m['stride']) == (0,1800,3)
    assert m['inference_kind'] == 'real' and m['pose_batch_size'] == 1
    assert m['device']['requested'] == 'cuda:0' and m['device']['cuda_available']
    assert m['pose_model']['model_id'] == 'usyd-community/vitpose-plus-base'
    assert m['pose_model']['resolved_revision'] == '92be54d7a29e42fad47b6e2ca01dd9e685a61e0d'
    assert m['pose_model']['dataset_index'] == 0
    smoke_manifest,smoke_rows = read_cache(ROOT/'.local/c2/vitpose_plus_20261109/observations_verified')
    assert m['pose_model']['files_sha256'] == smoke_manifest['pose_model']['files_sha256'], 'model artifacts differ from reviewed smoke'
    prefix_coordinate_delta,prefix_score_delta = 0.,0.
    for current,previous in zip(rows,smoke_rows):
        assert current['source_frame'] == previous['source_frame']
        for p,q in zip(current['persons'],previous['persons']):
            assert p['person_id'] == q['person_id'] and p['bbox_xywh'] == q['bbox_xywh']
            if p['pose_status'] == q['pose_status'] == 'ok':
                for a,b in zip(p['keypoints_xy_score'],q['keypoints_xy_score']):
                    prefix_coordinate_delta = max(prefix_coordinate_delta,abs(a[0]-b[0]),abs(a[1]-b[1]))
                    prefix_score_delta = max(prefix_score_delta,abs(a[2]-b[2]))
    assert m['video_sha256'] == file_sha256(video) == timing['video_sha256']
    assert m['input_sha256'] == meta['source_sha256'] == timing['annotation_metadata']['source_sha256']
    assert m['alignment_sha256'] == file_sha256(alignment)
    assert timing['decoder_verification']['verified_end_exclusive'] >= 1800
    assert timing['visual_review']['status'] == 'sampled-alignment-supported'
    assert (m['width'],m['height']) == (meta['width'],meta['height']) == (1920,1080)
    metrics_path = cache/'inference_metrics.json'
    assert file_sha256(metrics_path) == m['inference_metrics_sha256']
    metrics = json.loads(metrics_path.read_text(encoding='utf8'))
    counts,exclusions,groups = Counter(),Counter(),Counter()
    expected_calls = []
    min_score,max_score = float('inf'),float('-inf')
    low_scores = 0
    for row in rows:
        assert json.loads(json.dumps(row,allow_nan=False)) == row, 'raw JSON value round-trip changed observations'
        frame = row['source_frame']
        assert row['pts_ticks'] == timing['pts_ticks'][frame], 'source PTS mismatch'
        people = reference[frame]
        assert [p['person_id'] for p in row['persons']] == [p['person_id'] for p in people]
        for p,q in zip(row['persons'],people):
            assert p['bbox_xywh'] == q['bbox_xywh'], 'source box changed'
            assert p['detection_confidence'] is None, 'oracle box is not detector confidence'
            assert p['reference'] == {k:q[k] for k in ('outside','occluded','keyframe','attributes','box_metadata','track_metadata')}
            counts['input_records'] += 1
            counts['eligible'] += p['eligible']
            counts['outside'] += q['outside']
            counts['occluded'] += q['occluded']
            groups[q['attributes'].get('social_group','ATTRIBUTE_ABSENT')] += 1
            if p['eligible']:
                expected_calls.append((frame,p['person_id'],p['bbox_xywh']))
                counts['successful'] += p['pose_status'] == 'ok'
                counts['missing_eligible'] += p['pose_status'] == 'missing'
                if p['pose_status'] == 'ok':
                    # read_cache has already checked finite COCO17 xy/score arrays.
                    scores = [k[2] for k in p['keypoints_xy_score']]
                    min_score = min(min_score,min(scores));max_score = max(max_score,max(scores))
                    low_scores += sum(s < .6 for s in scores)
            else:
                assert p['pose_status'] == 'excluded' and p['keypoints_xy_score'] is None
                counts['excluded'] += 1
                exclusions[p['exclusion_reason']] += 1
    assert counts['input_records'] == counts['eligible']+counts['excluded']
    assert counts['eligible'] == counts['successful']+counts['missing_eligible']
    assert counts['successful'] == counts['eligible'], 'expected strict successful real pose calls'
    calls = metrics['calls']
    assert len(calls) == len(expected_calls)
    for call,(frame,pid,box) in zip(calls,expected_calls):
        assert call['source_frame'] == frame and call['person_ids'] == [pid] and call['input_bboxes_xywh'] == [box]
        assert call['input_shape'] == [1,3,256,192] and call['heatmap_shape'] == [1,17,64,48]
        assert call['output_shape'] == [1,17,3] and call['dataset_index'] == [0]
        assert call['labels_and_positional_association_verified'] and not call['hf_returned_bbox_used_for_association']
    forward = [c['forward_seconds'] for c in calls]
    complete_calls = [sum(c[k] for k in ('preprocess_seconds','forward_seconds','postprocess_seconds')) for c in calls]
    processing = metrics['decoded_and_pose_export_seconds']
    wall = run['wall_seconds']
    files = {p.name:{'bytes':p.stat().st_size,'sha256':file_sha256(p)} for p in cache.iterdir() if p.is_file()}
    return {'schema':'c2-full-cache-acceptance.v1','status':'PASS','cache':str(cache.resolve()),
        'frames':len(rows),'source_range':[0,1800],'stride':3,'json_value_roundtrip_verified':True,
        'accounting':dict(counts),
        'exclusions':dict(exclusions),'social_group_reference_counts':dict(groups),
        'model':m['pose_model'],'model_artifacts_match_reviewed_smoke':True,
        'smoke_prefix_comparison':{'max_coordinate_delta_pixels':prefix_coordinate_delta,
                                   'max_score_delta':prefix_score_delta},
        'environment':m['provenance'],'device':m['device'],
        'artifacts':files,'total_cache_file_bytes':sum(f['bytes'] for f in files.values()),
        'source_sha256':{'video':m['video_sha256'],'xml':m['input_sha256']},
        'score_diagnostics':{'range':[min_score,max_score],'scores_below_0_6':low_scores,
                             'scores_are_calibrated_probabilities':False},
        'performance':{'model_initialization_seconds':metrics['model_loading_and_download_seconds'],
            'process_wall_seconds':wall,'decoding_and_export_stage_seconds':processing,
            'decode_seconds':metrics['video_decode_seconds'],'observation_write_seconds':metrics['observation_record_write_seconds'],
            'model_forward_seconds_total':sum(forward),'forward_seconds_mean':statistics.mean(forward),
            'forward_seconds_median':statistics.median(forward),'person_call_seconds_mean':statistics.mean(complete_calls),
            'person_call_seconds_median':statistics.median(complete_calls),
            'preprocess_seconds_total':sum(c['preprocess_seconds'] for c in calls),
            'postprocess_seconds_total':sum(c['postprocess_seconds'] for c in calls),
            'persons_per_second_processing_stage':len(calls)/processing,
            'source_frames_per_second_processing_stage':1800/processing,
            'sampled_frames_per_second_processing_stage':600/processing,
            'persons_per_second_including_startup':len(calls)/wall,
            'source_frames_per_second_including_startup':1800/wall,
            'sampled_frames_per_second_including_startup':600/wall,
            'cuda_memory':metrics['cuda_memory'],'scope':'single measured oracle-box/oracle-identity run, not a robust benchmark'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache','video','xml','alignment','execution','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    if Path(sys.prefix).name != 'paper-research':
        raise SystemExit('Use paper-research explicitly')
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = validate_full(args.cache,args.video,args.xml,args.alignment,args.execution)
    with Path(args.output).open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','frames','accounting','performance','total_cache_file_bytes')},indent=2))
