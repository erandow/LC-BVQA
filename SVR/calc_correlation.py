from scipy.stats.mstats import spearmanr
from scipy.stats.mstats import pearsonr
import os


def calc_correlation(y_gt, y_pred, sc, dataset=None, delimiter="-", path=""):
    """Calculate SROCC with p and PLCC

    :param y_gt: Subjective MOSs of the validation or test set
    :param y_pred: predicted scores of the validation or test set
    :param sc: StandardScaler of the train set MOS
    :param dataset: if given, predicted scores have to be normalized to compare with cross dataset MOS
    :return: correlation between  MOSs and predicted scores
    """

    # Turn y_pred into a 2D array to match StandardScalar() input
    y_pred = y_pred.reshape(-1, 1)
    # Inverse transform the predicted values to get the real values
    y_pred = sc.inverse_transform(y_pred)

    # If dataset name has been given, normalize the results to compare with the cross dataset scores
    # since the ground-truth (y_gt) of the cross dataset has already been normalized
    if dataset == "konvid1k" or dataset == "youtube_ugc":
        y_pred -= 1
        y_pred /= 4.0
    elif dataset == "live_vqc":
        y_pred /= 100.0

    # Calculate the Spearman rank-order correlation
    srocc, p = spearmanr(y_gt.squeeze(), y_pred.squeeze())

    # Calculate the Pearson correlation
    plcc, _ = pearsonr(y_gt.squeeze(), y_pred.squeeze())

    text = f"Spearman correlation = {srocc:.4f} with p = {p:.4f},  Pearson correlation = {plcc:.4f}\n"
    with open(os.path.join(path, "correlation.txt"), "a") as writer:
        writer.write(text)
        writer.write(delimiter * 70 + "\n")

    return srocc, p, plcc
