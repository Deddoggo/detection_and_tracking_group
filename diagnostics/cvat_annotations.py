"""Convert manual CVAT XML/ZIP to original-frame evaluator annotations, no models.

Prefer CVAT for images 1.1 (dense, with track_id) for conversion. Native CVAT for
video 1.1 is also supported for axis-aligned rectangle tracks via linear keyframes.
"""
import argparse
import copy
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .core import digest, read, require_dense, same_source, validate, write

REVIEW_FIELDS=('all_frames_reviewed','all_visible_people_labeled','bbox_convention_confirmed',
               'ids_reviewed','interpolation_reviewed','source_alignment_checked')


def boolean(value):
    if str(value).lower() in ('true','1'):return True
    if str(value).lower() in ('false','0'):return False
    raise ValueError(f'invalid boolean {value!r}')


def load_xml(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith('annotations.xml')]
            if len(names)!=1:raise ValueError('Export ONE task, containing one annotations.xml')
            if z.getinfo(names[0]).file_size>32*1024*1024:raise ValueError('Unexpectedly large pilot export')
            data=z.read(names[0])
    else:
        data=Path(path).read_bytes()
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('DTD/entity declarations are not supported')
    root=ET.fromstring(data)
    if root.tag!='annotations' or root.findtext('version')!='1.1':
        raise ValueError('Expected CVAT XML format 1.1')
    return root


def attributes(box):
    return {a.attrib['name']:(a.text or '').strip() for a in box.findall('attribute')}


def shape(box, attrs):
    if float(box.get('rotation','0'))!=0:raise ValueError('Rotated boxes are not supported')
    xy=[float(box.attrib[k]) for k in ('xtl','ytl','xbr','ybr')]
    return {'xyxy':xy,'outside':boolean(box.get('outside','0')),
            'occluded':boolean(box.get('occluded','0')),'attributes':attrs,
            'keyframe':boolean(box.get('keyframe','1'))}


def expand_track(track, count):
    """Rectangle interpolation in FRAME space; attributes/occlusion hold from left.

    Outside starts at its keyframe; a last visible keyframe holds through clip end.
    No guess about identity/reappearance is made here.
    """
    entries=[];previous_attrs={}
    for box in sorted(track.findall('box'),key=lambda b:int(b.attrib['frame'])):
        frame=int(box.attrib['frame'])
        if frame<0 or frame>count or (frame==count and not boolean(box.get('outside','0'))):
            raise ValueError('track frame outside clip')
        if entries and frame==entries[-1][0]:raise ValueError('duplicate track keyframe')
        previous_attrs={**previous_attrs,**attributes(box)}
        entries.append((frame,shape(box,copy.deepcopy(previous_attrs))))
    if not entries:raise ValueError('non-rectangle/empty track')
    for index,(start,left) in enumerate(entries):
        end,right=entries[index+1] if index+1<len(entries) else (count,None)
        if left['outside']:continue
        for frame in range(start,min(end,count)):
            current=copy.deepcopy(left)
            if right is not None and frame>start:
                alpha=(frame-start)/(end-start)
                current['xyxy']=[a+(b-a)*alpha for a,b in zip(left['xyxy'],right['xyxy'])]
                current['keyframe']=False
            yield frame,current


def validate_mapping(mapping):
    count=mapping['clip']['frame_count'];start=mapping['source']['start_frame']
    if len(mapping['frames'])!=count or mapping['source']['end_frame_exclusive']-start!=count:
        raise ValueError('mapping count mismatch')
    if not mapping['verification']['all_decoded_pixel_hashes_match_source']:
        raise ValueError('clip/source frame preservation must be verified')
    for n,row in enumerate(mapping['frames']):
        if row['clip_frame']!=n or row['source_frame']!=start+n or row['mot_frame']!=n+1:
            raise ValueError('non-contiguous or invalid frame mapping')


def convert(root,mapping,review=None):
    validate_mapping(mapping)
    count=mapping['clip']['frame_count'];w=mapping['clip']['width'];h=mapping['clip']['height']
    task=root.find('meta/task')
    if task is None or int(task.findtext('size','-1'))!=count:
        raise ValueError('Task must contain every clip frame with frame step1; size mismatch')
    if task.findtext('flipped','False').lower()!='false':raise ValueError('Flipped task is unsupported')
    for field,expected in [('start_frame',0),('stop_frame',count-1)]:
        if task.findtext(field) is not None and int(task.findtext(field))!=expected:
            raise ValueError('Task frame range must cover full clip starting at zero')
    frame_filter=task.findtext('frame_filter','').strip()
    if frame_filter not in ('','step=1'):raise ValueError('Frame filtering/skipping is unsupported')
    width=task.findtext('original_size/width');height=task.findtext('original_size/height')
    if width is not None and (int(width),int(height))!=(w,h):raise ValueError('Task resolution mismatch')
    frames=[{'frame':r['source_frame'],'clip_frame':r['clip_frame'],
             'timestamp_seconds':r['source_timestamp_seconds'],'coverage':'pending','items':[]}
            for r in mapping['frames']]
    track_ids={};warnings=[];generated=0

    def add(n,s,track_id,label):
        if label!='person':raise ValueError(f'Unexpected label {label}; use only person')
        if s['outside']:return
        x1,y1,x2,y2=s['xyxy']
        if not all(math.isfinite(x) for x in (x1,y1,x2,y2)) or not 0<=x1<x2<=w or not 0<=y1<y2<=h:
            raise ValueError(f'Invalid/out-of-image bbox at clip frame {n}')
        a=s['attributes'];raw_id=a.get('person_id','')
        if not raw_id.isdecimal() or int(raw_id)<1:
            raise ValueError('Fill immutable person_id with positive integer on every track')
        pid=int(raw_id)
        if track_id in track_ids and track_ids[track_id]!=pid:raise ValueError('person_id changed within track')
        track_ids[track_id]=pid
        uncertain=boolean(a.get('identity_uncertain','false'))
        occlusion=a.get('occlusion','partial' if s['occluded'] else 'none')
        if occlusion not in ('none','partial','heavy'):raise ValueError('invalid occlusion category')
        if s['occluded']!=(occlusion!='none'):
            warnings.append(f'clip frame {n}, person {pid}: occluded flag and category disagree')
        item={'bbox':[x1,y1,x2-x1,y2-y1],'person_id':None if uncertain else pid,
              'candidate_person_id':pid,'annotation_track_id':str(track_id),
              'identity_uncertain':uncertain,'occlusion':occlusion,'occluded':s['occluded'],
              'truncated':boolean(a.get('truncated','false')),'ignore':False,
              'keyframe':s['keyframe'],'origin':'manual CVAT keyframe/interpolation'}
        # Visibility fraction is OPTIONAL; do not invent it from binary occlusion.
        if a.get('visibility') not in (None,'','unknown'):
            item['visibility']=float(a['visibility'])
        frames[n]['items'].append(item)

    images=root.findall('image');tracks=root.findall('track')
    if images and tracks:raise ValueError('Mixed image/track export unsupported')
    if images:
        numbers=[int(i.attrib['id']) for i in images]
        if len(set(numbers))!=count or set(numbers)!=set(range(count)) or len(numbers)!=count:
            raise ValueError('Dense image export must contain EVERY clip frame, including empty frames')
        for image in images:
            n=int(image.attrib['id'])
            if (int(image.attrib['width']),int(image.attrib['height']))!=(w,h):raise ValueError('image resolution mismatch')
            for box in image.findall('box'):
                if box.get('source','manual') not in ('manual','semi-auto'):
                    raise ValueError('This pilot accepts manual/interpolated boxes only')
                a=attributes(box)
                if 'track_id' not in a:raise ValueError('Use Track mode: dense export needs track_id attribute')
                add(n,shape(box,a),a['track_id'],box.attrib['label'])
            if any(e.tag not in ('box','tag') for e in image):raise ValueError('Only rectangle person tracks supported')
        export_mode='CVAT for images dense; interpolation already resolved by CVAT'
    else:
        for track in tracks:
            if track.get('source','manual') not in ('manual','semi-auto'):
                raise ValueError('This pilot accepts manual/interpolated annotations only')
            if any(e.tag!='box' for e in track):raise ValueError('Only axis-aligned rectangles supported')
            for n,s in expand_track(track,count):
                generated+=int(not s['keyframe'])
                add(n,s,track.attrib['id'],track.attrib['label'])
        export_mode='CVAT for video keyframes expanded linearly; cross-check against CVAT dense export'
        if not tracks:warnings.append('No tracks in export; clip contains people so verify incomplete annotation')
    for f in frames:
        candidate_ids=[i['candidate_person_id'] for i in f['items']]
        if len(set(candidate_ids))!=len(candidate_ids):raise ValueError('same person_id assigned to two boxes in one frame')
    reviewed=bool(review and all(review.get(k) is True for k in REVIEW_FIELDS) and
                  review.get('reviewer') and review.get('review_date'))
    if reviewed:
        for f in frames:f['coverage']='full'
    doc={'schema_version':1,'kind':'annotations','source':copy.deepcopy(mapping['source']),
         'annotation_status':'human_reviewed' if reviewed else 'pending',
         'annotator':review.get('reviewer') if review else None,'review':review,
         'clip':mapping['clip'],'frames':frames,
         'conversion':{'export_mode':export_mode,'generated_interpolated_boxes':generated,
                       'track_to_person_id':track_ids,'warnings':sorted(set(warnings)),
                       'bbox_convention':'visible person extent, pixel xywh; no amodal reconstruction'}}
    return validate(doc)


def qa(doc):
    validate(doc)
    frames=doc['frames'];items=[i for f in frames for i in f['items']]
    ids=sorted({i['person_id'] for i in items if i['person_id'] is not None})
    tracking_error=None
    try:require_dense(doc,annotated=True)
    except ValueError as exc:tracking_error=str(exc)
    empty=[f.get('clip_frame',f['frame']) for f in frames if not f['items']]
    warnings=doc.get('conversion',{}).get('warnings',[])
    return {'frames':len(frames),'boxes':len(items),'resolved_person_ids':ids,
            'uncertain_identity_boxes':sum(i.get('identity_uncertain',False) for i in items),
            'empty_frames_to_review':empty,'warnings':warnings,
            'annotation_status':doc['annotation_status'],
            'tracking_export_ready':tracking_error is None and not warnings,
            'tracking_export_blocker':tracking_error,
            'human_completeness_proven_by_script':False,'ground_truth_metrics':None}


def preview(doc,mapping,out_dir,ffmpeg,clip_frames):
    """Draw exported labels on exact clip frames; no inference."""
    import subprocess
    from PIL import Image,ImageDraw
    validate(doc);validate_mapping(mapping);same_source(doc,{'source':mapping['source']})
    clip=Path(mapping['clip']['path'])
    if digest(clip)!=mapping['clip']['sha256']:raise ValueError('annotation clip hash mismatch')
    out=Path(out_dir)
    if out.exists():raise FileExistsError('Choose a new preview folder')
    out.mkdir(parents=True)
    records={f['frame']:f for f in doc['frames']}
    for n in clip_frames:
        if not 0<=n<mapping['clip']['frame_count']:raise ValueError('clip frame outside range')
        raw=out/f'frame_{n:06d}.png'
        subprocess.run([ffmpeg,'-v','error','-nostdin','-threads','2','-i',str(clip),
                        '-vf',f'select=eq(n\\,{n})','-frames:v','1','-fps_mode','passthrough',
                        '-threads','2',str(raw)],check=True)
        with Image.open(raw) as image:
            draw=ImageDraw.Draw(image)
            source_frame=mapping['frames'][n]['source_frame']
            for i in records[source_frame]['items']:
                x,y,w,h=i['bbox'];color='orange' if i.get('identity_uncertain') else 'lime'
                draw.rectangle([x,y,x+w,y+h],outline=color,width=2)
                label=f'ID {i["person_id"]}' if i['person_id'] is not None else f'ID ? ({i["candidate_person_id"]})'
                draw.text((x,max(0,y-12)),label,fill=color,stroke_width=1,stroke_fill='black')
            draw.text((8,8),f'clip frame {n} | source frame {source_frame} | manual annotations',
                      fill='yellow',stroke_width=1,stroke_fill='black')
            image.save(out/f'review_{n:06d}.png')
        raw.unlink()  # Only our just-created temporary frame.


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('convert');c.add_argument('--export',required=True);c.add_argument('--mapping',required=True)
    c.add_argument('--review');c.add_argument('--out-dir',required=True)
    c=sub.add_parser('check');c.add_argument('--input',required=True);c.add_argument('--report')
    c=sub.add_parser('make-review');c.add_argument('--export',required=True);c.add_argument('--mapping',required=True);c.add_argument('--out',required=True)
    c=sub.add_parser('preview');c.add_argument('--input',required=True);c.add_argument('--mapping',required=True)
    c.add_argument('--out-dir',required=True);c.add_argument('--ffmpeg',required=True);c.add_argument('--frames',nargs='+',type=int,required=True)
    a=p.parse_args()
    if a.command=='preview':
        preview(read(a.input),read(a.mapping),a.out_dir,a.ffmpeg,a.frames);return
    if a.command=='make-review':
        write(a.out,{'reviewer':None,'review_date':None,'mapping_sha256':digest(a.mapping),
                     'export_sha256':digest(a.export),**{k:False for k in REVIEW_FIELDS}})
        return
    if a.command=='check':
        report=qa(read(a.input));print(json.dumps(report,indent=2))
        if a.report:write(a.report,report)
        return
    mapping=read(a.mapping);review=read(a.review) if a.review else None
    if review and review.get('mapping_sha256')!=digest(a.mapping):raise ValueError('Review refers to different mapping')
    if review and review.get('export_sha256')!=digest(a.export):raise ValueError('Review refers to different export; recheck after edits')
    if Path(a.out_dir).exists():raise FileExistsError('Choose a fresh conversion folder')
    doc=convert(load_xml(a.export),mapping,review)
    doc['conversion'].update(export_sha256=digest(a.export),mapping_sha256=digest(a.mapping))
    report=qa(doc)
    write(Path(a.out_dir)/'annotations.json',doc);write(Path(a.out_dir)/'qa_report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
