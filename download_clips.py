import argparse
import multiprocessing as mp
import os
from time import time as timer
import glob

import subprocess
from tqdm import tqdm
from functools import partial
from vfhq_dl.parse_meta_info import parse_clip_meta

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='meta_info',
                    help='Directory containing clip meta files')
parser.add_argument('--output_dir', type=str, default='data/youtube_videos',
                    help='Location to download videos')
parser.add_argument('--num_workers', type=int, default=8,
                    help='How many multiprocessing workers?')
args = parser.parse_args()

def download_video(output_dir, video_id):
    r"""Download video."""
    video_path = os.path.join(output_dir, video_id)
    if not os.path.isfile(video_path):
        try:
            # Download the highest quality mp4 stream.
            command = [
                "yt-dlp",
                "https://youtube.com/watch?v={}".format(video_id), "--quiet", "-f",
                "bestvideo*+bestaudio",
                "--output", video_path,
                "--no-continue"
            ]
            return_code = subprocess.call(command)
            success = return_code == 0

        except Exception as e:
            print(e)
            print('Failed to download %s' % (video_id))
    else:
        print('File exists: %s' % (video_id))

if __name__ == '__main__':
    # Read list of videos.
    meta_fnames = glob.glob(os.path.join(args.input_dir, '*.txt'))

    clip_metas = []
    for fname in meta_fnames:
        clip_metas.append(parse_clip_meta(fname))

    # calculate median end_t - start_t
    durations = [meta.end_t - meta.start_t for meta in clip_metas]
    median_duration = sorted(durations)[len(durations) // 2]
    print('Median duration: %.2f' % (median_duration))

    video_ids = set(meta.video_id for meta in clip_metas)
    print('Found %d unique videos' % (len(video_ids)))

    # Create output folder.
    os.makedirs(args.output_dir, exist_ok=True)

    # # Download videos.
    downloader = partial(download_video, args.output_dir)

    start = timer()
    pool_size = args.num_workers
    print('Using pool size of %d' % (pool_size))
    with mp.Pool(processes=pool_size) as p:
        _ = list(tqdm(p.imap_unordered(downloader, video_ids), total=len(video_ids)))
    print('Elapsed time: %.2f' % (timer() - start))