from create_dataset import create_dataset
from train import train
import os
import warnings

warnings.filterwarnings("ignore")


def main(
    features_path,
    mos_path,
    layers,
    kernel="rbf",
    pooling_methods=["avg", "max"],
    video_label="video",
    mos_label="mos",
):
    _, X, y = create_dataset(
        features_path, mos_path, layers, pooling_methods, video_label, mos_label
    )
    path = f".\\results\\{'_'.join(layers)}_{kernel}"
    os.makedirs(path, exist_ok=True)
    train(X, y, kernel, path=path)


if __name__ == "__main__":
    features_path = "D:\\IPRIA\\LiveVQA\\Pooled Features"
    mos_path = "E:\\IPRIA Datasets\\LiveVQC\\mos.csv"
    layers = [
        # ["inception.avgpool", "inception.Mixed_5c.cat", "inception.Mixed_5b.cat"],
        ["inception.avgpool", "inception.Mixed_5c.cat"],
        # ["inception.avgpool", "inception.Mixed_5b.cat"],
        # ["inception.avgpool"],
        # ["inception.Mixed_5c.cat", "inception.Mixed_5b.cat"],
        # ["inception.Mixed_5c.cat"],
        # ["inception.Mixed_5b.cat"],
    ]
    kernel = ["linear"]

    for layer in layers:
        for k in kernel:
            print("Training on", layer, k)
            main(
                features_path=features_path,
                mos_path=mos_path,
                layers=layer,
                kernel=k,
                video_label="flickr_id",
                mos_label="mos",
            )
