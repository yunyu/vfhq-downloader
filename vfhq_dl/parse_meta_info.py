from typing import NamedTuple
import os

class ClipMeta(NamedTuple):
    video_id: str
    pid: int
    clip_idx: int
    frame_start: int
    frame_end: int
    start_t: float
    end_t: float
    duration_t: float
    height: int
    width: int
    fps: float
    x0: int
    y0: int
    x1: int
    y1: int


def parse_clip_meta(clip_meta_path):
    # read the basic info
    clip_meta_file = open(clip_meta_path, "r")
    clip_name = os.path.splitext(os.path.basename(clip_meta_path))[0]
    for line in clip_meta_file:
        if line.startswith("H"):
            clip_height = int(line.strip().split(" ")[-1])
        if line.startswith("W"):
            clip_width = int(line.strip().split(" ")[-1])
        if line.startswith("FPS"):
            clip_fps = float(line.strip().split(" ")[-1])
        # get the coordinates of face
        if line.startswith("CROP"):
            clip_crop_bbox = line.strip().split(" ")[-4:]
            x0 = int(clip_crop_bbox[0])
            y0 = int(clip_crop_bbox[1])
            x1 = int(clip_crop_bbox[2])
            y1 = int(clip_crop_bbox[3])

    _, videoid, pid, clip_idx, frame_rlt = clip_name.split("+")
    pid = int(pid.split("P")[1])
    clip_idx = int(clip_idx.split("C")[1])
    frame_start, frame_end = frame_rlt.replace("F", "").split("-")
    # NOTE
    frame_start, frame_end = int(frame_start) + 1, int(frame_end) - 1

    start_t = round(frame_start / float(clip_fps), 5)
    end_t = round(frame_end / float(clip_fps), 5)
    duration_t = end_t - start_t

    return ClipMeta(
        pid=pid,
        video_id=videoid,
        clip_idx=clip_idx,
        frame_start=frame_start,
        frame_end=frame_end,
        start_t=start_t,
        end_t=end_t,
        duration_t=duration_t,
        height=clip_height,
        width=clip_width,
        fps=clip_fps,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
    )
