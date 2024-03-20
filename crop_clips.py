import argparse
import multiprocessing as mp
import os
from time import time as timer
import glob

import subprocess
from tqdm import tqdm
from functools import partial
from vfhq_dl.parse_meta_info import parse_clip_meta, ClipMeta
from vfhq_dl.util import VIDEO_EXTENSIONS
from decimal import Decimal, DivisionByZero
import ffmpeg
import functools

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='meta_info',
                    help='Directory containing clip meta files')
parser.add_argument('--download_dir', type=str, default='data/youtube_videos',
                    help='Location of downloaded videos')
parser.add_argument('--output_dir', type=str, default='data/cropped_videos',
                    help='Location of cropped videos')
parser.add_argument('--num_workers', type=int, default=8,
                    help='How many multiprocessing workers?')
args = parser.parse_args()


@functools.lru_cache(maxsize=2048)
def get_h_w_fps(filepath):
    probe = ffmpeg.probe(filepath)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    height = int(video_stream['height'])
    width = int(video_stream['width'])
    
    # Extract avg_frame_rate and convert to Decimal FPS
    avg_frame_rate = video_stream['avg_frame_rate']
    numerator, denominator = map(int, avg_frame_rate.split('/'))
    if denominator != 0:  # Prevent division by zero
        fps = Decimal(numerator) / Decimal(denominator)
    else:
        fps = Decimal(0)  # Handle division by zero, if applicable
    
    return height, width, fps

def format_ts(ts_seconds) -> str:
    # Convert the total seconds into hours, minutes, and seconds
    hours, remainder = divmod(ts_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format the timestamp string
    timestamp = "{:02}:{:02}:{:06.3f}".format(int(hours), int(minutes), float(seconds))
    
    return timestamp

def trim_and_crop(download_dir: str, output_dir: str, clip_meta: ClipMeta):
    output_filename = f"Clip+{clip_meta.video_id}+P{clip_meta.pid}+C{clip_meta.clip_idx}.mp4"
    output_filepath = os.path.join(output_dir, output_filename)

    if os.path.exists(output_filepath):
        print('Output file %s exists, skipping' % (output_filepath))
        return

    input_filepath = os.path.join(download_dir, clip_meta.video_id)
    for ext in VIDEO_EXTENSIONS:
        if os.path.exists(input_filepath + ext):
            input_filepath += ext
            break

    if not os.path.exists(input_filepath):
        print('Input file %s does not exist, skipping' % (input_filepath))
        return

    h, w, fps = get_h_w_fps(input_filepath)
    H, W, L, T, R, B = clip_meta.height, clip_meta.width, clip_meta.x0, clip_meta.y0, clip_meta.x1, clip_meta.y1
    
    if clip_meta.end_t - clip_meta.start_t < 4:
        print('Clip under 4 seconds, skipping:', output_filename)
        return

    t = int(T / H * h)
    b = int(B / H * h)
    l = int(L / W * w)
    r = int(R / W * w)

    stream = ffmpeg.input(input_filepath, ss=format_ts(clip_meta.start_t), to=format_ts(clip_meta.end_t))
    video = stream.video
    audio = stream.audio
    video = ffmpeg.crop(video, l, t, r-l, b-t)
    stream = ffmpeg.output(
        audio,
        video,
        output_filepath,
        **{
            "c:v": "h264_nvenc",
            "preset": "slow",
            "b:v": "0",
            "cq:v": "24",
            "rc:v": "vbr",
        }
    )
    args = stream.get_args()
    command = ["ffmpeg", "-loglevel", "quiet"] + args
    return_code = subprocess.call(command)
    success = return_code == 0
    if not success:
        print('Command failed:', command)

if __name__ == '__main__':
    # Read list of videos.
    meta_fnames = glob.glob(os.path.join(args.input_dir, '*.txt'))

    clip_metas = []
    for fname in meta_fnames:
        clip_metas.append(parse_clip_meta(fname))

    print('Found %d unique clips' % (len(clip_metas)))

    # Create output folder.
    os.makedirs(args.output_dir, exist_ok=True)

    # # Download videos.
    downloader = partial(trim_and_crop, args.download_dir, args.output_dir)

    start = timer()
    pool_size = args.num_workers
    print('Using pool size of %d' % (pool_size))
    with mp.Pool(processes=pool_size) as p:
        _ = list(tqdm(p.imap_unordered(downloader, clip_metas), total=len(clip_metas)))
    print('Elapsed time: %.2f' % (timer() - start))