import torch
import torch.nn as nn
from torchvision import models
import yaml


class InceptionV3Regressor(nn.Module):
    def __init__(self):
        super(InceptionV3Regressor, self).__init__()
        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT
        )
        self.inception.fc = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.inception(x)
        return x

    def unfreeze_last_layers(self, layers_to_unfreeze):
        print("Layers", layers_to_unfreeze)
        for param in self.inception.parameters():
            param.requires_grad = False
        for layer_name in layers_to_unfreeze:
            layer = getattr(self.inception, layer_name)
            for param in layer.parameters():
                param.requires_grad = True


class EfficientNetRegressor(nn.Module):

    def __init__(self):
        super(EfficientNetRegressor, self).__init__()

        # Load the EfficientNet-B0 model pre-trained on ImageNet
        self.efficientnet = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )

        # Modify the classifier to output a single value for regression
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(self.efficientnet.classifier[1].in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.efficientnet(x)
        return x

    def unfreeze_last_layers(self, layers_to_unfreeze):
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        for layer_name in layers_to_unfreeze:
            layer = getattr(self.efficientnet, layer_name)
            for param in layer.parameters():
                param.requires_grad = True


class VGGRegressor(nn.Module):

    def __init__(self):
        super(VGGRegressor, self).__init__()

        # Load the VGG16 model pre-trained on ImageNet
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Modify the classifier to output a single value for regression
        self.vgg.classifier[6] = nn.Sequential(
            nn.Linear(4096, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 1)
        )
        self.freeze_all()

    def forward(self, x):
        x = self.vgg(x)
        return x

    def unfreeze_last_layers(self, layers_to_unfreeze):
        for param in self.vgg.parameters():
            param.requires_grad = False
        for layer_name in layers_to_unfreeze:
            layer = getattr(self.vgg, layer_name)
            for param in layer.parameters():
                param.requires_grad = True


def get_regressor_model(model_name):
    if model_name == "vgg":
        return VGGRegressor()
    elif model_name == "efficientnet":
        return EfficientNetRegressor()
    elif model_name == "inception":
        return InceptionV3Regressor()
    else:
        raise ValueError(
            f"Model name '{model_name}' is not recognized. Available models are: 'vgg', 'efficientnet', 'inception'."
        )


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config
