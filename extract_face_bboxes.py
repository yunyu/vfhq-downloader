import argparse
import asyncio
import glob
import multiprocessing as mp
import os
from functools import partial
from time import time as timer

import torch
from tqdm import tqdm
import numpy as np

from vfhq_dl.video_util import extract_bboxes_for_video

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cropped_video_dir",
    type=str,
    default="data/cropped_videos",
    help="Location of cropped videos",
)
parser.add_argument("--save_dir", type=str, default="data/video_bboxes")
parser.add_argument(
    "--num_workers", type=int, default=2, help="How many multiprocessing workers?"
)
args = parser.parse_args()


def run_extraction_and_save_bboxes(save_dir, video_path):
    video_basename = os.path.basename(video_path)
    video_fname, _ = os.path.splitext(video_basename)
    out_fname = os.path.join(save_dir, video_fname + ".npz")

    if os.path.exists(out_fname):
        print("File already exists: ", out_fname)
        return

    print("Processing: ", video_path)

    try: 
        bboxes, vid_dims = extract_bboxes_for_video(video_path)
        np.savez(out_fname, bboxes=bboxes, vid_dims=vid_dims)
        print("Saved to: ", out_fname)
    except Exception as e:
        print(f"Error processing video {video_basenam}: {e}")


if __name__ == "__main__":
    fnames = glob.glob(os.path.join(args.cropped_video_dir, "*.mp4"))

    print("Found %d unique clips" % (len(fnames)))

    # delete output file if it exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    extractor = partial(run_extraction_and_save_bboxes, args.save_dir)
    start = timer()

    pool_size = args.num_workers
    print("Using pool size of %d" % (pool_size))
    with mp.get_context('spawn').Pool(processes=pool_size) as p:
        _ = list(tqdm(p.imap_unordered(extractor, fnames), total=len(fnames)))

    print("Elapsed time: %.2f" % (timer() - start))
