from sklearn.preprocessing import StandardScaler
from plot_correlation import plot_correlation
from calc_correlation import calc_correlation
import joblib
import os
from create_dataset import create_dataset


if __name__ == "__main__":
    features_path = "D:\\IPRIA\\LiveVQA\\Pooled Features"
    mos_path = "E:\\IPRIA Datasets\\LiveVQC\\mos.csv"
    model_path = "D:\\IPRIA\\KoNViD-1k\\results\\inception.avgpool_inception.Mixed_5c.cat_linear\\best_svr_model.joblib"

    layers = ["inception.avgpool", "inception.Mixed_5c.cat"]
    pooling_methods = ["avg", "max"]
    video_label = "flickr_id"
    mos_label = "mos"
    _, X, y = create_dataset(
        features_path, mos_path, layers, pooling_methods, video_label, mos_label
    )

    if not os.path.exists(model_path):
        print("Model does not exist")
        exit()
    model = joblib.load(model_path)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)

    y_pred = model.predict(X)

    srocc, p, plcc = calc_correlation(y, y_pred, sc_y)

    plot_correlation(y, y_pred, sc_y, len(y), srocc)
