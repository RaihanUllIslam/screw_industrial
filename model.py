import torch
from torchvision.models import resnet18, ResNet18_Weights

class PretrainedModel:
    def __init__(self, num_classes=2, freeze=False):
        self.num_classes = num_classes
        self.freeze = freeze
        self.model = self.initialize_model()

    def initialize_model(self):
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, self.num_classes)

        if self.freeze:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

        return model
