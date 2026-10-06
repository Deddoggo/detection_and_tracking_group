"""Render exported human labels on source frames; no inference."""
import argparse
import subprocess
from pathlib import Path
from .core import digest, read, validate


def main():
    from PIL import Image, ImageDraw, ImageFont
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--ffmpeg',required=True)
    p.add_argument('--frames',nargs='+',type=int,required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();doc=validate(read(a.input));source=doc['source'];out=Path(a.out_dir)
    if doc['kind']!='annotations':raise ValueError('Human annotations required')
    if digest(source['video'])!=source['video_sha256']:raise ValueError('Source hash mismatch')
    if out.exists():raise FileExistsError('Choose a new preview folder')
    out.mkdir(parents=True);records={f['frame']:f for f in doc['frames']}
    font=ImageFont.load_default(size=16)
    for n in a.frames:
        if n not in records:raise ValueError('Frame outside converted range')
        raw=out/f'raw_{n}.png'
        subprocess.run([a.ffmpeg,'-v','error','-nostdin','-threads','2','-i',source['video'],
                        '-vf',f'select=eq(n\\,{n})','-frames:v','1','-fps_mode','passthrough',str(raw)],check=True)
        with Image.open(raw) as image:
            draw=ImageDraw.Draw(image)
            for item in records[n]['items']:
                x,y,w,h=item['bbox'];color='orange' if item['identity_uncertain'] else 'lime'
                draw.rectangle((x,y,x+w,y+h),outline=color,width=2)
                draw.text((x,max(0,y-17)),item['annotation_track_id'],fill=color,font=font,stroke_width=1,stroke_fill='black')
            draw.text((10,10),f'Source/CVAT frame {n} | native CVAT IDs | exported human labels',font=font,fill='yellow',stroke_width=1,stroke_fill='black')
            image.save(out/f'review_{n:06d}.png')
        raw.unlink()


if __name__=='__main__':main()
