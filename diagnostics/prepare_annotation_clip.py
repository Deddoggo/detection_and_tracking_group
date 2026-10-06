"""Prepare an exact-frame, lossless 15s annotation clip and verified source mapping."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

from .core import digest, read, write


def hashes(ffmpeg, args):
    result = subprocess.check_output([ffmpeg, '-v', 'error', '-nostdin', *args,
                                      '-map', '0:v:0', '-an', '-fps_mode', 'passthrough', '-f', 'framemd5', '-'], text=True)
    return [line.split(',')[-1].strip() for line in result.splitlines()
            if line and not line.startswith('#')]


def prepare(timestamp_file, source_descriptor, target, ffmpeg, ffprobe):
    ts, source = read(timestamp_file), read(source_descriptor)
    video = Path(source['video'])
    if digest(video) != ts['video_sha256'] or source['video_sha256'] != ts['video_sha256']:
        raise ValueError('original video hash does not match timestamps/source')
    start, end = source['start_frame'], source['end_frame_exclusive']
    count = end-start
    out = Path(target)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh annotation folder')
    out.mkdir(parents=True,exist_ok=True)
    clip = out/'pilot01_30-45s.unverified.mp4'
    # Uniform presentation clock only: preserve every original frame one-to-one.
    select = f'select=between(n\\,{start}\\,{end-1})'
    # Raw frame pipe deliberately removes the VFR clock before encoding. Using a
    # sync filter directly can duplicate/drop a frame even when the count matches.
    decoder_command=[ffmpeg,'-v','error','-nostdin','-threads','4','-i',str(video),
        '-map','0:v:0','-an','-vf',select,'-frames:v',str(count),'-fps_mode','passthrough',
        '-pix_fmt','yuvj420p','-threads','4','-f','rawvideo','pipe:1']
    command=[ffmpeg,'-v','error','-nostdin','-f','rawvideo','-pixel_format','yuv420p',
        '-video_size',f'{source["width"]}x{source["height"]}','-framerate',f'{count}/15',
        '-color_range','pc','-i','pipe:0','-an','-frames:v',str(count),
        '-c:v','libx264','-preset','fast','-crf','0','-color_range','pc',
        '-threads','4','-video_track_timescale',str(count*100),'-movflags','+faststart',str(clip)]
    decoder=subprocess.Popen(decoder_command,stdout=subprocess.PIPE)
    try:
        subprocess.run(command,stdin=decoder.stdout,check=True)
    finally:
        decoder.stdout.close()
    if decoder.wait()!=0:raise ValueError('source frame extraction failed')
    probe = json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0',
        '-show_frames','-show_entries','frame=best_effort_timestamp_time:stream=width,height,nb_frames,avg_frame_rate:format=duration',
        '-of','json',str(clip)],text=True))
    original_hashes = hashes(ffmpeg,['-threads','4','-i',str(video),'-vf',select,
                                    '-frames:v',str(count),'-threads','4'])
    clip_hashes = hashes(ffmpeg,['-threads','4','-i',str(clip),'-threads','4'])
    if len(probe['frames']) != count or len(original_hashes) != count or original_hashes != clip_hashes:
        raise ValueError('Frame-preservation verification failed; do not annotate this clip')
    width,height=probe['streams'][0]['width'],probe['streams'][0]['height']
    if (width,height)!=(source['width'],source['height']):
        raise ValueError('resolution changed')
    verified_clip=out/'pilot01_30-45s.mp4'
    clip.rename(verified_clip);clip=verified_clip
    rows = [{'clip_frame':i,'mot_frame':i+1,'clip_timestamp_seconds':float(f['best_effort_timestamp_time']),
             'source_frame':start+i,'source_timestamp_seconds':ts['frames'][start+i]['timestamp_seconds'],
             'decoded_pixel_md5':clip_hashes[i]} for i,f in enumerate(probe['frames'])]
    mapping={'schema_version':1,'clip':{'path':str(clip.resolve()),'sha256':digest(clip),
             'frame_count':count,'width':width,'height':height,'duration_seconds':float(probe['format']['duration']),
             'presentation_fps':f'{count}/15'},'source':source,'frames':rows,
             'timing_policy':'one original decoded frame per clip frame; uniform 15s presentation; use ORIGINAL timestamps for analysis',
             'verification':{'all_decoded_pixel_hashes_match_source':True,'frames_dropped':0,'frames_duplicated':0,
                             'ffmpeg_commands':[decoder_command,command],'model_inference':False}}
    write(out/'frame_mapping.json',mapping)
    with (out/'frame_mapping.csv').open('w',newline='',encoding='utf8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    write(out/'source.json',source)
    print(json.dumps({'clip':str(clip),'frame_count':count,'duration_seconds':mapping['clip']['duration_seconds'],
                      'all_pixels_preserved':True}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('timestamps','source','out-dir','ffmpeg','ffprobe'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a.timestamps,a.source,a.out_dir,a.ffmpeg,a.ffprobe)


if __name__=='__main__':main()
