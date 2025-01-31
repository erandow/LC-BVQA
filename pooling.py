from tqdm import tqdm
from os.path import join

import numpy as np
import pandas as pd

import torch
import os


def pooling(
    features_path, pooled_path, feature_names, selected_frames_path=None, method="avg"
):
    """
    Pool the features with different strategies such as avg and max

    The given directory must be as follows:
    [video name] > [extracted layer names] > [frames].pt

    for example:
    2999049224/avgpool/0.pt
    """

    # To know which video features are pooled
    features_videos = os.listdir(features_path)

    for video_index in tqdm(range(len(features_videos))):
        video = features_videos[video_index]

        video_layers_features_path = os.path.join(features_path, video)
        video_layer_features = os.listdir(video_layers_features_path)

        selected_frames = np.arange(
            len(
                os.listdir(
                    os.path.join(video_layers_features_path, video_layer_features[0])
                )
            )
        )

        if selected_frames_path:
            selected_frames = (
                pd.read_csv(os.path.join(selected_frames_path, str(video) + ".csv"))
                .to_numpy(dtype="int")
                .flatten()
            )

        dist_path = os.path.join(pooled_path, method, video)

        for layer_feature in feature_names:
            frame_features_path = os.path.join(
                video_layers_features_path, layer_feature
            )
            frame_features = os.listdir(frame_features_path)

            all_frames_features = []

            for frame_index in selected_frames:
                frame_feature_path = os.path.join(
                    frame_features_path, frame_features[frame_index]
                )
                frame_feature_tensor = torch.load(frame_feature_path)
                all_frames_features.append(frame_feature_tensor)

            all_frames_tensors = torch.stack(all_frames_features)

            os.makedirs(dist_path, exist_ok=True)

            if method == "max":
                pooled, _ = torch.max(all_frames_tensors, dim=0)
            else:
                pooled = torch.mean(all_frames_tensors, dim=0)

            torch.save(pooled, os.path.join(dist_path, layer_feature + ".pt"))


pooling(
    "D:\\IPRIA\\KoNViD-1k\\Spatial Features",
    "D:\\IPRIA\\KoNViD-1k\\Pooled Features",
    ["inception.avgpool", "inception.Mixed_5b.cat", "inception.Mixed_5c.cat"],
    method="avg",
)
pooling(
    "D:\\IPRIA\\KoNViD-1k\\Spatial Features",
    "D:\\IPRIA\\KoNViD-1k\\Pooled Features",
    ["inception.avgpool", "inception.Mixed_5b.cat", "inception.Mixed_5c.cat"],
    method="max",
)
