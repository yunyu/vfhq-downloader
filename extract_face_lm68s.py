import argparse
import asyncio
import glob
import multiprocessing as mp
import os
from functools import partial
from time import time as timer
import decord

import face_alignment
import numpy as np
import torch
from decord import VideoReader, cpu, gpu
from tqdm import tqdm

from vfhq_dl.video_util import ctx, VIDEO_HEIGHT, VIDEO_WIDTH

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cropped_video_dir",
    type=str,
    default="data/cropped_videos",
    help="Location of cropped videos",
)
parser.add_argument("--save_dir", type=str, default="data/video_lm68s")
parser.add_argument(
    "--num_workers", type=int, default=2, help="How many multiprocessing workers?"
)
args = parser.parse_args()

fa = None

def extract_lm68s_for_video(video_path):
    global fa
    if fa is None:
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda"
        )
        pass

    vr = VideoReader(video_path, ctx=ctx, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
    num_frames = len(vr)
    batch_size = 32

    def get_biggest_detected_face_idxes(bboxes):
        idxes: list[int] = []
        for curr_frame in bboxes:
            if len(curr_frame) == 0:
                idxes.append(-1)
                continue

            biggest_face_idx = 0
            biggest_face_size = 0

            for i, face in enumerate(curr_frame):
                x1, y1, x2, y2, _conf = face
                face_size = (x2 - x1) * (y2 - y1)
                if face_size > biggest_face_size:
                    biggest_face_size = face_size
                    biggest_face_idx = i

            idxes.append(biggest_face_idx)
        return idxes

    lms = []

    for batch_idx in range(0, num_frames, batch_size):
        frames = vr.get_batch(
            list(range(batch_idx, min(batch_idx + batch_size, num_frames)))
        ).asnumpy()
        preds, _, bboxes = fa.get_landmarks_from_batch(frames, return_bboxes=True)
        biggest_face_idxes = get_biggest_detected_face_idxes(bboxes)
        for curr_frame_lms, biggest_face_idx in zip(preds, biggest_face_idxes):
            if biggest_face_idx == -1:
                # use the last frame's landmarks
                lms.append(lms[-1])
                continue
            lms.append(curr_frame_lms[biggest_face_idx * 68: (biggest_face_idx + 1) * 68])

    lms = np.array(lms)
    vid_dims = np.array([VIDEO_WIDTH, VIDEO_HEIGHT])

    return lms, vid_dims


def run_extraction_and_save_lm68s(save_dir, video_path):
    video_basename = os.path.basename(video_path)
    video_fname, _ = os.path.splitext(video_basename)
    out_fname = os.path.join(save_dir, video_fname + ".npz")

    if os.path.exists(out_fname):
        print("File already exists: ", out_fname)
        return

    print("Processing: ", video_path)

    # try:
    lm68s, vid_dims = extract_lm68s_for_video(video_path)
    np.savez(out_fname, lm68s=lm68s, vid_dims=vid_dims)
    print("Saved to: ", out_fname)
    # except Exception as e:
    #     print(f"Error processing video {video_basename}: {e}")


if __name__ == "__main__":
    fnames = glob.glob(os.path.join(args.cropped_video_dir, "*.mp4"))

    print("Found %d unique clips" % (len(fnames)))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    extractor = partial(run_extraction_and_save_lm68s, args.save_dir)
    start = timer()

    pool_size = args.num_workers
    print("Using pool size of %d" % (pool_size))
    with mp.get_context('spawn').Pool(processes=pool_size) as p:
        _ = list(tqdm(p.imap_unordered(extractor, fnames), total=len(fnames)))

    print("Elapsed time: %.2f" % (timer() - start))
