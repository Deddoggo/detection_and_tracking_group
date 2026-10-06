"""Combine raw-detection counts and official tracking summary, or write pending table."""
import argparse
import csv
import math
from pathlib import Path
from .core import digest, read, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detection');p.add_argument('--tracking-summary')
    p.add_argument('--out-dir',required=True)
    a=p.parse_args();out=Path(a.out_dir)
    if out.exists():raise FileExistsError('Choose a new report folder')
    if bool(a.detection)!=bool(a.tracking_summary):raise ValueError('Provide both outputs or neither for a pending table')
    names=['Precision','Recall','FP','FN','Duplicates','HOTA','DetA','AssA','IDF1','IDSW','Fragmentation']
    vals={n:None for n in names};provenance={}
    if a.detection:
        d=read(a.detection)
        for display,key in [('Precision','precision'),('Recall','recall'),('FP','fp'),('FN','fn'),('Duplicates','duplicates')]:
            vals[display]=d[key]
        with Path(a.tracking_summary).open(encoding='utf8') as f:
            rows=[r for r in csv.reader(f,delimiter=' ') if r]
        if len(rows)!=2 or len(rows[0])!=len(rows[1]):raise ValueError('Expected official two-row summary')
        t=dict(zip(rows[0],rows[1]))
        for key in ['HOTA','DetA','AssA','IDF1','IDSW','Frag']:
            n=float(t[key])
            if not math.isfinite(n):raise ValueError('Non-finite tracking result')
            vals['Fragmentation' if key=='Frag' else key]=int(n) if key in ('IDSW','Frag') else n
        provenance={'detection_sha256':digest(a.detection),'tracking_summary_sha256':digest(a.tracking_summary),
                    'requirement':'summary must contain exactly this one verified sequence; verify evaluation manifest/seqmap'}
    write(out/'baseline_table.json',{'status':'results supplied' if a.detection else 'pending predictions and verified GT',
        'scope':'single diagnostic/development sequence; no generalization claim','values':vals,'provenance':provenance})
    lines=['# Diagnostic baseline','', '| Metric | Value |','|---|---|']
    for n in names:
        v=vals[n]
        if v is None: label='Pending'
        elif n in ('Precision','Recall'):label=f'{100*v:.3f}%'
        elif n in ('HOTA','DetA','AssA','IDF1'):label=f'{v:.3f} (official summary scale 0–100)'
        else:label=str(v)
        lines.append(f'| {n} | {label} |')
    lines.extend(['','Detection: detector observations before tracking, IoU 0.5 one-to-one matching.',
                  'Tracking: official HOTA/CLEAR/Identity; Frag is fragmentation, not IDSW.',
                  'Verify the scored range, source hash, annotation review and evaluator sanity checks before interpreting results.',
                  'No method improvement has been selected from a pending table.'])
    (out/'baseline_table.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':main()
