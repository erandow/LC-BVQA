from PIL import Image
import numpy as np
import os

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Dropout, ReLU

from torchvision import datasets, transforms, models
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from torchvision import models

from feature_extraction.models import InceptionV3Regressor

import yaml

from tqdm import tqdm

from flopth import flopth


def isRepetetive(features_path, name):
    return name + ".csv" in os.listdir(features_path)


def get_top_triangle(matrix):
    indices = np.triu_indices(matrix.shape[0])
    return matrix[indices]


def gram_matrix(input):
    a, b, c, d = (
        input.size()
    )  # a=batch size(=1), b=number of feature maps, (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
    G = torch.matmul(features, features.t())  # compute the gram product
    return G.div(a * b * c * d)


def flatten_symmetric(matrix):
    # Get the indices of the upper triangular part of the matrix
    row_indices, col_indices = torch.triu_indices(matrix.size(0), matrix.size(1))

    # Get the values of the upper triangular part of the matrix
    upper_triangular_part = matrix[row_indices, col_indices]
    return upper_triangular_part


def extract_spatial_features(
    videos_path,
    features_path,
    model,
    feature_extractor,
    image_transforms,
    device,
    isGram=True,
):
    video_names = os.listdir(videos_path)
    skipped = []
    for index in range(len(video_names)):
        video_name = video_names[index]
        print(index + 1)
        video_path = os.path.join(videos_path, video_name)
        video_features_path = os.path.join(features_path, video_name)

        if os.path.exists(video_features_path):
            skipped.append(video_name)
            continue

        frame_names = os.listdir(video_path)

        for frame_index in range(len(frame_names)):
            frame_name = frame_names[frame_index]
            with torch.no_grad():
                frame_path = os.path.join(video_path, frame_name)
                frame = image_transforms(Image.open(frame_path))[None, :].to(device)
                features = feature_extractor(frame)

                for feature_layer in features:
                    video_directory = os.path.join(video_features_path, feature_layer)
                    #                     video_directory = os.path.join(video_features_path)
                    path = os.path.join(video_directory, str(frame_index) + ".pt")

                    os.makedirs(video_directory, exist_ok=True)

                    if feature_layer == "inception.avgpool":
                        torch.save(features[feature_layer].flatten(), path)
                    else:
                        upper_triangle_gram = flatten_symmetric(
                            gram_matrix(features[feature_layer])
                        )
                        torch.save(upper_triangle_gram, path)

    print("Skipped List: ", len(skipped))


def get_model(model_path, device):
    model = InceptionV3Regressor()

    if model_path:
        model.load_state_dict(torch.load(model_path))
    else:
        print("Creating raw model...")

    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    print("Model is ready!!!")
    return model


def extract_frame_features():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The process is starting on this device: ", device)

    model_path = (
        "C:\\Users\\easad\\Desktop\\Feature Extraction\\6\\trained_model_SGD_3.pth"
    )
    model = get_model(model_path, device)
    layers_name = [
        "inception.Mixed_5b.cat",
        "inception.Mixed_5c.cat",
        "inception.avgpool",
    ]
    feature_extractor = create_feature_extractor(model, return_nodes=list(layers_name))

    image_transforms = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    frames_path = "E:\\IPRIA Datasets\\konvid1k\\frames"
    features_path = "D:\\IPRIA\\KoNViD-1k\\Spatial Features"

    extract_spatial_features(
        frames_path,
        features_path,
        model,
        feature_extractor,
        image_transforms,
        device=device,
    )


extract_frame_features()
