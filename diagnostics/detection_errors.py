"""Export inspection candidates from GT matching; no inference, no direction claims."""
import argparse
import collections
from .core import evaluate_detection, match, read, write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--detections',required=True)
    p.add_argument('--annotations',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();d=read(a.detections);g=read(a.annotations)
    summary=evaluate_detection(d,g,.5);pred={f['frame']:f['items'] for f in d['frames']}
    rows=[];strata=collections.defaultdict(lambda:{'gt':0,'matched':0,'fn':0})
    for f in g['frames']:
        gt=f['items'];ps=pred[f['frame']];pairs=match(ps,gt,.5)
        used_p={p for p,_ in pairs};used_g={n for _,n in pairs}
        fp=[i for n,i in enumerate(ps) if n not in used_p]
        fn=[i for n,i in enumerate(gt) if n not in used_g]
        rows.append({'frame':f['frame'],'fp':len(fp),'fn':len(fn),'fp_boxes':fp,'fn_boxes':fn})
        for n,item in enumerate(gt):
            height=item['bbox'][3]
            for key in [('height_lt50' if height<50 else 'height_50_to99' if height<100 else 'height_ge100'),
                        'occluded' if item.get('occluded') else 'not_occluded']:
                strata[key]['gt']+=1;strata[key]['matched']+=int(n in used_g);strata[key]['fn']+=int(n not in used_g)
    for s in strata.values():s['recall']=s['matched']/s['gt'] if s['gt'] else None
    write(a.out,{'summary':summary,'descriptive_gt_strata':dict(strata),
                 'worst_30_frames':sorted(rows,key=lambda x:(-(x['fp']+x['fn']),x['frame']))[:30],
                 'interpretation':'GT bbox height is annotation-dependent; descriptive strata, not validated causal attribution',
                 'next_decision':'inspect error candidates and official ByteTrack versus stitched metrics before choosing one improvement'})


if __name__=='__main__':main()
