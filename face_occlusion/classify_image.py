from torchvision import transforms
import torch

# CONSTANT
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = [224, 224]
CLASSES = {0: "non-occluded",
           1: "occluded"}

transform = transforms.Compose([
            transforms.Resize(SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
    ])


def classify_image(img, model, device):
    img = transform(img).to(device)
    output = model(img.unsqueeze(0))
    output = torch.softmax(output, 1)
    prob, pred = torch.max(output, 1)
    return CLASSES[pred.item()], prob.item() * 100
