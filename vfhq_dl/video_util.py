from decord import VideoReader
from decord import cpu, gpu
import numpy as np
from PIL import Image
import face_detection
import torch

ctx = cpu(0)
VIDEO_WIDTH, VIDEO_HEIGHT = 512, 512

fa = None
                                
def sample_frames_from_video(video_path: str):
    vr = VideoReader(video_path, ctx=ctx, width=512, height=512)

    out = []
    for i in range(3):
        try:
            out.append(Image.fromarray(vr.next().asnumpy()))
        except StopIteration:
            break
        vr.skip_frames(20)

    return out

def get_all_frames_from_video(video_path: str, video_width: int, video_height: int):
    vr = VideoReader(video_path, ctx=ctx, width=video_width, height=video_height)
    # TODO: Check BGR (opencv) vs RGB
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    return frames

def extract_bboxes_for_video(video_path):
    global fa
    if fa is None:
        fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
									device='cuda')

    vr = VideoReader(video_path, ctx=ctx, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
    num_frames = len(vr)
    batch_size = 32
    bboxes = []

    for batch_idx in range(0, num_frames, batch_size):
        frames = vr.get_batch(list(range(batch_idx, min(batch_idx+batch_size, num_frames)))).asnumpy()
        # print(video_path, frames.shape)
        preds = fa.get_detections_for_batch(frames)
        bboxes.extend(preds)

    bboxes = np.array(bboxes)
    vid_dims = np.array([VIDEO_WIDTH, VIDEO_HEIGHT])

    return bboxes, vid_dims