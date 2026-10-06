"""Check official metric behavior on known synthetic cases before scoring GT."""
import argparse
import sys
from pathlib import Path
from .core import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trackeval-root',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(a.trackeval_root).resolve()))
    import numpy as np
    import trackeval
    results={}
    for name,ids in [('perfect',[0,0,0]),('middle_miss',[0,None,0]),('one_switch',[0,1,1])]:
        data={'num_gt_ids':1,'num_tracker_ids':max(x for x in ids if x is not None)+1,
              'num_gt_dets':3,'num_tracker_dets':sum(x is not None for x in ids),'num_timesteps':3,
              'gt_ids':[np.array([0]) for _ in ids],
              'tracker_ids':[np.array([] if x is None else [x],dtype=int) for x in ids],
              'similarity_scores':[np.empty((1,0)) if x is None else np.array([[1.]]) for x in ids]}
        h=trackeval.metrics.HOTA().eval_sequence(data)
        c=trackeval.metrics.CLEAR({'THRESHOLD':.5,'PRINT_CONFIG':False}).eval_sequence(data)
        i=trackeval.metrics.Identity({'THRESHOLD':.5,'PRINT_CONFIG':False}).eval_sequence(data)
        expected={'perfect':(1.,1.,0,0),'middle_miss':(2/3,.8,0,1),'one_switch':((5/9)**.5,2/3,1,0)}[name]
        actual=(float(np.mean(h['HOTA'])),float(i['IDF1']),int(c['IDSW']),int(c['Frag']))
        if not np.allclose(actual,expected):raise AssertionError((name,actual,expected))
        results[name]={'HOTA':actual[0],'IDF1':actual[1],'IDSW':actual[2],'Frag':actual[3]}
    from .core import freeze
    write(a.out,{'status':'passed synthetic official metric checks; not playground metrics',
                 'cases':results,'provenance':freeze({},[trackeval.__file__]),
                 'remaining':'MOT adapter must also load the exact scored range without exclusions/errors'})


if __name__=='__main__':main()
