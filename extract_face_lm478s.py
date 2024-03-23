import argparse
import glob
import multiprocessing as mp
import os
from functools import partial
from time import time as timer

import numpy as np
from tqdm import tqdm

from face_landmarking.face_landmarker import MediapipeLandmarker
from vfhq_dl.video_util import VIDEO_HEIGHT, VIDEO_WIDTH, get_all_frames_from_video

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cropped_video_dir",
    type=str,
    default="data/cropped_videos",
    help="Location of cropped videos",
)
parser.add_argument("--save_dir", type=str, default="data/video_lm478s")
parser.add_argument(
    "--num_workers", type=int, default=6, help="How many multiprocessing workers?"
)
args = parser.parse_args()

face_landmarker = None


def extract_and_save_lm478s(save_dir, video_path):
    global face_landmarker
    if face_landmarker is None:
        face_landmarker = MediapipeLandmarker(
            read_video_to_frames=lambda video_path: get_all_frames_from_video(
                video_path, VIDEO_WIDTH, VIDEO_HEIGHT
            )
        )

    video_basename = os.path.basename(video_path)
    video_fname, _ = os.path.splitext(video_basename)
    out_fname = os.path.join(save_dir, video_fname + ".npz")

    if os.path.exists(out_fname):
        print("File already exists: ", out_fname)
        return

    print("Processing: ", video_path)

    try:
        img_lm478, vid_lm478 = face_landmarker.extract_lm478_from_video_name(video_path)
        lm478 = face_landmarker.combine_vid_img_lm478_to_lm478(img_lm478, vid_lm478)
        vid_dims = np.array([VIDEO_WIDTH, VIDEO_HEIGHT])
        np.savez(out_fname, lm478s=lm478, vid_dims=vid_dims)
        print("Saved to:", out_fname)
    except Exception as e:
        print(f"Error processing video {video_basename}: {e}")


if __name__ == "__main__":
    fnames = glob.glob(os.path.join(args.cropped_video_dir, "*.mp4"))
    print("Found %d unique clips" % (len(fnames)))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    extractor = partial(extract_and_save_lm478s, args.save_dir)
    start = timer()

    pool_size = args.num_workers
    print("Using pool size of %d" % (pool_size))
    with mp.get_context("spawn").Pool(processes=pool_size) as p:
        _ = list(tqdm(p.imap_unordered(extractor, fnames), total=len(fnames)))

    print("Elapsed time: %.2f" % (timer() - start))
