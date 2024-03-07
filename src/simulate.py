import numpy as np
from typing import Tuple

def simulate_data(n:int=100, p:int=10, random_seed:int=0) ->  Tuple[np.ndarray, np.ndarray, float]:

    # Simulate the logistic regression parameters and data
    rng = np.random.RandomState(random_seed)
    X = rng.normal(size =(n, p)).round(2)
    beta = np.zeros(p)
    beta[0] = 1
    beta[0] = 2
    eta = -2 + X@beta
    p = 1/(1+np.exp(-eta))
    y = rng.binomial(1, p)

    # Now, we will add some missing values to the data
    missing = rng.binomial(1, 0.1, size = X.shape).astype(bool)
    X[missing] = np.nan

    # Compute the brier loss between p and y -- the smallest it could be
    brier_loss = np.mean((p-y)**2)
    return X, y, brier_loss

