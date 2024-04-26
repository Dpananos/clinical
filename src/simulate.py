import numpy as np
import pandas as pd
from typing import Tuple
from scipy.special import logit

def simulate_data(n:int=100, p:int=10, random_seed:int=0) ->  Tuple[np.ndarray, np.ndarray, float]:

    # Simulate the logistic regression parameters and data
    rng = np.random.RandomState(random_seed)
    X = rng.normal(size =(n, p)).round(2)
    beta = np.zeros(p)
    beta[0] = 3
    beta[1] = 3
    eta = -2 + X@beta
    p = 1/(1+np.exp(-eta))
    y = rng.binomial(1, p)

    # Now, we will add some missing values to the data
    missing = rng.binomial(1, 0.1, size = X.shape).astype(bool)
    X[missing] = np.nan

    # Compute the brier loss between p and y -- the smallest it could be
    brier_loss = np.mean((p-y)**2)
    return X, y, brier_loss


def simulate_non_linear_data(n:int=100, p:int=10, random_seed:int=0) ->  Tuple[np.ndarray, np.ndarray, float]:

    # Simulate the logistic regression parameters and data
    rng = np.random.RandomState(random_seed)
    X = rng.normal(size =(n, p)).round(2)
    beta = np.zeros(p)
    beta[0] = 3
    beta[1] = 3
    eta = -2 + X@beta - X[:, 0]**2
    p = 1/(1+np.exp(-eta))
    y = rng.binomial(1, p)

    # Now, we will add some missing values to the data
    missing = rng.binomial(1, 0.1, size = X.shape).astype(bool)
    X[missing] = np.nan

    # Compute the brier loss between p and y -- the smallest it could be
    brier_loss = np.mean((p-y)**2)
    return X, y, brier_loss


def simulate_simple_data(n:int=100, random_seed:int=0) ->  Tuple[np.ndarray, np.ndarray, float]:

    # Simulate the logistic regression parameters and data
    rng = np.random.RandomState(random_seed)
    X = rng.gamma(shape=1, scale=1, size=n)
    b0 = logit(0.05)
    b1 = logit(0.1) - b0
    eta = b0 + b1*X
    p = 1/(1+np.exp(-eta))
    y = rng.binomial(1, p)

    return X.reshape(-1, 1), y


