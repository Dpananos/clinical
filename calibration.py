from src.simulate import simulate_simple_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, KFold
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.compose import ColumnTransformer
from statsmodels.nonparametric.smoothers_lowess import lowess
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score


def fit_model(resample_minority: bool = True) -> None:
    X, y = simulate_simple_data(100_000, 0)
    Xtest, ytest = simulate_simple_data(10_000, 1)

    if resample_minority:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

    clf = LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=1000,
        learning_rate=0.005,
        verbose=-1
    )

    # Grid for cross validation
    param_grid = {
        'max_depth': [2, 4, 6],
        "lambda": [0.1, 1, 2],
    }

    model = GridSearchCV(clf, param_grid, cv=3, scoring="neg_log_loss",verbose=4)

    model.fit(X, y)

    predicted_p = model.predict_proba(Xtest)

    auc = roc_auc_score(ytest, predicted_p[:, 1])

    return ytest, predicted_p[:, 1], auc

if __name__ == "__main__":

    fig, axes = plt.subplots(nrows=1, ncols=2, dpi=120)

    for ax, rsmp in zip(axes, [True, False]):
        ytest, predicted_p, auc = fit_model(resample_minority=rsmp)

        prange = np.linspace(min(predicted_p), max(predicted_p), 25)
        apparent_cal = lowess(ytest, predicted_p, it=0, xvals=prange)


        #plot calibration curve
        ax.scatter(predicted_p, ytest, s=1, alpha = 0.4, c='k')
        ax.plot(prange, apparent_cal, 'red', label = 'Non-Parametric Estimate')
        ax.plot([0, 1], [0, 1], 'k--', label = 'Perfect Calibration')
        ax.set_title(f"AUC: {auc:.2f}, Resampled with SMOTE: {str(rsmp)}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Outcome")
        ax.legend(loc = 'upper left')
    
    plt.show()

