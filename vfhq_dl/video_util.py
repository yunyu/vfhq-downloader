from decord import VideoReader
from decord import cpu, gpu
import numpy as np
from PIL import Image

ctx = cpu(0)

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