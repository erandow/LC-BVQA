from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from plot_correlation import plot_correlation
from calc_correlation import calc_correlation


def train(X, y, kernel="rbf", epochs=20, n_splits=5):
    SROCC_coef, SROCC_p, PLCC = [], [], []
    best_acc = 0
    best_model = None

    # Feature Scaling
    sc_X = StandardScaler()
    sc_y = StandardScaler()

    for _ in tqdm(range(epochs)):
        # Use K-Fold Cross Validation for evaluation
        kfold = KFold(n_splits=n_splits, shuffle=True)

        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)
            y_train = sc_y.fit_transform(y_train)

            regressor = SVR(kernel=kernel)
            regressor.fit(X_train, y_train.squeeze())

            # Predict the scores for X_test videos features
            y_pred = regressor.predict(X_test)

            srocc, p, plcc = calc_correlation(y_test, y_pred, sc_y)
            if plcc > best_acc:
                best_acc = plcc
                best_model = regressor

            SROCC_coef.append(srocc)
            SROCC_p.append(p)
            PLCC.append(plcc)

            plot_correlation(y_test, y_pred, sc_y, len(SROCC_coef), srocc)
