from face_occlusion.model import Model as OcclusionModel, load_weight
from vfhq_dl.video_util import sample_frames_from_video
from face_occlusion.classify_image import classify_image
import torch

model = OcclusionModel("vgg16", 2, False).to("cuda")
model = load_weight(model, "pretrained_models/vgg16_occlusion/best_vgg16.pth")
model.eval()

o = sample_frames_from_video("data/cropped_videos/Clip+-aQ4eQV8uH0+P0+C2.mp4")

for img in o:
    print(classify_image(img, model, device="cuda"))