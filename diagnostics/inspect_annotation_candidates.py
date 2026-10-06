"""Produce source-frame crop contact sheets for manual inspection, no inference."""
import argparse
import subprocess
from pathlib import Path
from .core import digest, read


def main():
    from PIL import Image,ImageDraw,ImageFont
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',required=True);p.add_argument('--timestamps',required=True)
    p.add_argument('--ffmpeg',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--frames',type=int,nargs='+',required=True)
    a=p.parse_args();doc=read(a.annotations);timing=read(a.timestamps);out=Path(a.out_dir)
    if out.exists():raise FileExistsError('Choose fresh inspection folder')
    if digest(timing['video'])!=timing['video_sha256']:raise ValueError('Source hash mismatch')
    if a.frames!=sorted(set(a.frames)) or any(n<0 or n>=len(timing['frames']) for n in a.frames):
        raise ValueError('Select distinct ordered source frames')
    out.mkdir(parents=True);expr='+'.join(f'eq(n\\,{n})' for n in a.frames)
    subprocess.run([a.ffmpeg,'-v','error','-nostdin','-threads','2','-i',timing['video'],
                    '-vf','select='+expr,'-frames:v',str(len(a.frames)),'-fps_mode','passthrough',
                    str(out/'source_%03d.png')],check=True)
    records={f['frame']:f['items'] for f in doc['frames']};font=ImageFont.load_default(size=16)
    regions={'left_tree':(540,450,780,730),'bottom_right':(1490,820,1920,1020)}
    for label,region in regions.items():
        cw,ch=region[2]-region[0],region[3]-region[1]
        cols=4;sheet=Image.new('RGB',(cols*cw,((len(a.frames)+cols-1)//cols)*(ch+28)),(35,35,35))
        for i,n in enumerate(a.frames):
            with Image.open(out/f'source_{i+1:03d}.png') as image:
                draw=ImageDraw.Draw(image)
                for item in records.get(n,[]):
                    x,y,w,h=item['bbox'];draw.rectangle((x,y,x+w,y+h),outline='lime',width=2)
                    draw.text((x,max(0,y-16)),item['annotation_track_id'],font=font,fill='lime',stroke_width=1,stroke_fill='black')
                crop=image.crop(region)
                x=(i%cols)*cw;y=(i//cols)*(ch+28)
                sheet.paste(crop,(x,y+28))
                ImageDraw.Draw(sheet).text((x+4,y+4),f'{n} | {timing["frames"][n]["timestamp_seconds"]:.3f}s',font=font,fill='white')
        sheet.save(out/f'{label}_contact_sheet.png')
    print('Manual candidate inspection only; frames beyond reviewed boundary are context, not GT')


if __name__=='__main__':main()
