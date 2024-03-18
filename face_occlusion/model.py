from torch import nn
from PIL import ImageFile
from torchvision import models
from torch import save, load


ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_weight(model, file, show=True):
    checkpoints = load(file)
    if show:
        print("Model at epoch:", checkpoints["epoch"])
    model.load_state_dict(checkpoints["state_dict"])
    return model


def get_model(name, pretrained):
    model = getattr(models, name)(weights=get_pretrained(name) if pretrained else None)
    return model


class Model(nn.Module):

    def __init__(
        self, name: str, num_class: int, pretrained: bool = False, is_train: bool = True
    ):
        super(Model, self).__init__()

        self.model = get_model(name, pretrained)

        # Change the number of class
        if "resnet" in name:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_class)
        elif "densenet" in name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_class)
        elif "vgg" in name:
            in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features, num_class)
        elif "convnext" in name:
            in_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Linear(in_features, num_class)
        if is_train:
            print(f"Model: {name}")

    def forward(self, x):
        return self.model(x)
