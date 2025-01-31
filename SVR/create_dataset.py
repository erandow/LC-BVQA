import os

import pandas as pd
import numpy as np
import torch


def create_dataset(
    features_path,
    mos_path,
    selected_layers,
    pooling_methods=["avg", "max"],
    video_label="flickr_id",
    mos_label="mos",
):
    """The structure of the features mus\be like below
    [video name] > [layer_name].pt
    """
    if len(selected_layers) <= 0:
        raise "There should exist at leasr one selected layer"
    if len(pooling_methods) <= 0:
        raise "There should exist at least one pooling method"

    mos_df = pd.read_csv(mos_path)
    video_names = os.listdir(os.path.join(features_path, pooling_methods[0]))

    X = []
    y = []
    names = []

    for video_name in video_names:
        feature_data = np.array([])
        for pooling_method in pooling_methods:
            for index, layer in enumerate(selected_layers):
                feature = torch.load(
                    os.path.join(
                        features_path, pooling_method, video_name, layer + ".pt"
                    )
                )
                feature_data = np.concatenate(
                    (feature_data, feature.flatten().cpu().numpy())
                )

        video_mos = mos_df.loc[mos_df[video_label] == video_name, mos_label].values[0]

        names.append(video_name)
        X.append(feature_data)
        y.append(video_mos)

    return np.array(names), np.array(X), np.array(y).reshape(-1, 1)
