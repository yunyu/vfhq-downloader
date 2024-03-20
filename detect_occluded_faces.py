from vfhq_dl.video_util import sample_frames_from_video
from vfhq_dl.classify_face_occluded import classify_face_occluded
import torch
import glob
import os
import argparse
import tqdm
import asyncio

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cropped_video_dir",
    type=str,
    default="data/cropped_videos",
    help="Location of cropped videos",
)
parser.add_argument("--output_file", type=str, default="data/occluded_videos.txt")
args = parser.parse_args()

async def main():
    fnames = tqdm.tqdm(glob.glob(os.path.join(args.cropped_video_dir, "*.mp4")))
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

    # delete output file if it exists
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    for fname in fnames:
        try:
            o = sample_frames_from_video(fname)
        except Exception as e:
            print(f"Error reading video: {e}")
            continue
        # classify second extracted frame
        occluded = await classify_face_occluded(o[1])
        if occluded:
            print(f"Occluded: {fname}")
            with open(args.output_file, "a") as f:
                f.write(fname + "\n")

if __name__ == "__main__":
    asyncio.run(main())