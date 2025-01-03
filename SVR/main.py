from create_dataset import create_dataset
from train import train


def main(features_path, mos_path, layers, kernel="rbf"):
    names, X, y = create_dataset(features_path, mos_path, layers)
    train(X, y, kernel)


if __name__ == "__main__":
    features_path = "D:\\IPRIA\\LiveVQA\\Pooled Features"
    mos_path = "E:\\LIVE Video Quality Challenge (VQC) Database_2\\LIVE Video Quality Challenge (VQC) Database\\mos.csv"
    layers = ["inception.avgpool", "inception.Mixed_5b.cat", "inception.Mixed_5c.cat"]

    main(features_path=features_path, mos_path=mos_path, layers=layers)
