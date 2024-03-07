from src.simulate import simulate_data

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, KFold
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.compose import ColumnTransformer


def fit_model(short: bool = True) -> None:
    X, y, true_brier_loss = simulate_data(1000, 10, 0)

    # Create a column transformer to model the first feature as a Spline
    # and the rest as a standard scaler
    column_transformer = ColumnTransformer(
        [("spline", SplineTransformer(), [0])], remainder="passthrough"
    )

    # Set the pipeline.  This will impute, scale and estimate the model
    # all on one step.
    # We can also loop over several values of the hyperparameters
    estimation_steps = Pipeline(
        [
            ("imputer", KNNImputer()),
            ("column_transformer", column_transformer),
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=10_000)),
        ]
    )

    # Define grid of hyperparameters over which to search
    grid = {
        "column_transformer__spline__n_knots": [3, 5, 7, 9],
        "imputer__n_neighbors": [1, 3, 5, 7, 9],
        "logistic__C": [0.01, 0.1, 1, 10, 100],
        "logistic__penalty": ["l2"],
        "logistic__solver": ["saga"],
    }

    brier_scorer = make_scorer(
        brier_score_loss, greater_is_better=False, needs_proba=True
    )

    if short:
        cv_inner = RepeatedKFold(n_splits=3, n_repeats=3)
        cv_outer = KFold(n_splits=3)
        verbose = 0
    else:
        cv_inner = RepeatedKFold(n_splits=10, n_repeats=100)
        cv_outer = KFold(n_splits=10)
        verbose = 2

    model = GridSearchCV(
        estimator=estimation_steps,
        param_grid=grid,
        scoring=brier_scorer,
        cv=cv_inner,
        verbose=verbose,
        n_jobs=-1,
    )

    # Scores are negative so that sklearn can optimize them,
    # Just multiply by negative 1
    scores = -1.0 * cross_val_score(model, X, y, cv=cv_outer, scoring=brier_scorer)
    print(f"Estimated Brier Loss: {scores.mean():.3f} ")
    print(f"True Brier Loss: {true_brier_loss:.3f} ")


if __name__ == "__main__":
    fit_model(short=True)
