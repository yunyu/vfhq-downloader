import argparse
import asyncio
import glob
import multiprocessing as mp
import os
from functools import partial
from time import time as timer

import torch
from tqdm import tqdm

from vfhq_dl.classify_face_occluded import classify_face_occluded
from vfhq_dl.video_util import sample_frames_from_video

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cropped_video_dir",
    type=str,
    default="data/cropped_videos",
    help="Location of cropped videos",
)
parser.add_argument("--output_file", type=str, default="data/occluded_videos.txt")
parser.add_argument(
    "--num_workers", type=int, default=4, help="How many multiprocessing workers?"
)
args = parser.parse_args()

l = mp.Lock()

def detect_occlusion_for_file(output_file: str, fname: str):
    try:
        o = sample_frames_from_video(fname)
    except Exception as e:
        print(f"Error reading video: {e}")
        return
    occluded = classify_face_occluded(o[1])
    if occluded:
        print(f"Occluded: {fname}")
        l.acquire()
        with open(args.output_file, "a") as f:
            f.write(fname + "\n")
        l.release()

if __name__ == "__main__":
    fnames = glob.glob(os.path.join(args.cropped_video_dir, "*.mp4"))
    # fnames = [
    #     os.path.join(args.cropped_video_dir, fname)
    #     for fname in [
    #         "Clip+79vu9mgSorY+P0+C0.mp4",
    #         "Clip+-aQ4eQV8uH0+P0+C2.mp4",
    #         "Clip+_Xf0vkqPWzg+P0+C0.mp4",
    #         "Clip+jwuL71fWom0+P0+C0.mp4",
    #         "Clip+an23gYUOdSY+P0+C0.mp4",
    #         "Clip+Z2_OJWthxbA+P0+C0.mp4",
    #         "Clip+TMgDFDn_vDg+P0+C0.mp4",
    #         "Clip+7p-3VN1_bpw+P0+C2.mp4"
    #     ]
    # ]

    print("Found %d unique clips" % (len(fnames)))

    # delete output file if it exists
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    detector = partial(detect_occlusion_for_file, args.output_file)
    start = timer()
    pool_size = args.num_workers
    print("Using pool size of %d" % (pool_size))
    with mp.Pool(processes=pool_size) as p:
        _ = list(tqdm(p.imap_unordered(detector, fnames), total=len(fnames)))
    print("Elapsed time: %.2f" % (timer() - start))
