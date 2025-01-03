def plot_correlation(y_gt, y_pred, sc, num, srocc, dataset=None, cross_dataset=None):
    """Plot SROCC of predicted and ground-truth scores

    :param y_gt: Subjective MOSs of the validation or test set
    :param y_pred: predicted scores of the validation or test set
    :param sc: StandardScaler of the train set MOS
    :param num: number indicating the time SVR is trained and also the plot number
    :param srocc: correlation to be included in the plot file name
    :param dataset: if plot is intended for cross dataset, it used to normalize the predicted scores
    :param cross_dataset: if given, normalized scores are brought back the cross dataset MOS range
    """
    # Turn y_pred into a 2D array to match StandardScalar() input
    y_pred = y_pred.reshape(-1, 1)
    # Inverse transform the predicted values to get the real values
    y_pred = sc.inverse_transform(y_pred)

    if cross_dataset is not None:
        if dataset == "live_vqc":
            y_pred /= 100.0
        elif dataset == "konvid1k" or dataset == "youtube_ugc":
            y_pred -= 1
            y_pred /= 4

        if cross_dataset == "live_vqc":
            y_pred *= 100
            file_path = f"plots/{num}_{abs(srocc):.4f}_{cross_dataset}.png"
        elif cross_dataset == "konvid1k" or cross_dataset == "youtube_ugc":
            y_pred *= 4
            y_pred += 1
            file_path = f"plots/{num}_{abs(srocc):.4f}_{cross_dataset}.png"
    else:
        file_path = f"plots/{num}_{abs(srocc):.4f}.png"

    # Plot the correlation between ground-truth and predicted scores
    sns.set(style="darkgrid")
    scatter_plot = sns.relplot(
        x=y_gt.squeeze(),
        y=y_pred.squeeze(),
        kind="scatter",
        height=7,
        aspect=1.2,
        palette="coolwarm",
    ).set(xlabel="Ground-truth MOS", ylabel="Predicted Score")

    plt.close()
    scatter_plot.savefig(file_path)
