"""M1-M4 regression functions and data generation.

Parameters
----------
model : str
    Model name: M1, M2, M3, or M4.
X : array
    Covariates at which the regression function is evaluated.
n : int
    Number of labeled observations, with both covariates and responses.
N : int
    Number of unlabeled observations, with covariates only.
d : int
    Number of covariates.
seed : int
    Random seed for data generation.
noise_sigma : float
    Standard deviation of the normal error added to the response.

Functions
---------
regression_function computes the model values at X.
projection_target gives the exact beta, with the intercept first.
generate draws independent uniform covariates on [0, 1] and normal errors.
It returns X, y, X_u, and the model values at X_u.
"""

import numpy as np

MODELS = ("M1", "M2", "M3", "M4")


def regression_function(model, X):
    if model == "M1":
        product = np.prod(2.0 / 3.0 + 2.0 * X * (1.0 - X), axis=1)
        return 10.0 * np.sum(X**2, axis=1) + 10.0 * product
    if model == "M2":
        product = np.prod(2.0 / 3.0 + 2.0 * X * (1.0 - X), axis=1)
        J = 3.0
        return 20.0 * J**-2.0 * np.sum(np.sin(2.0 * np.pi * J * X), axis=1) + 4.0 * product
    if model == "M3":
        return 5.0 * np.sum(X**4, axis=1)
    if model == "M4":
        # sum over pairs j < k of P2(x_j) P2(x_k), P2 the shifted Legendre polynomial
        p2 = 6.0 * X**2 - 6.0 * X + 1.0
        pairs = 0.5 * (np.sum(p2, axis=1) ** 2 - np.sum(p2**2, axis=1))
        return 5.0 * np.sum(X**4, axis=1) + 3.0 * pairs
    raise ValueError(f"unknown model {model!r}")


def projection_target(model, d):
    """beta0 minimizing E(m(X) - b0 - X'b)^2, in closed form."""
    if model == "M1":
        # Each 10 x^2 term projects to 10 x - 5/3.
        # The product has mean 1 and zero slope in the linear projection.
        return np.concatenate(([10.0 - 5.0 * d / 3.0], np.full(d, 10.0)))
    if model == "M2":
        slope = -40.0 / (9.0 * np.pi)  # (20/9) Cov(sin(6 pi x), x) / Var(x)
        return np.concatenate(([4.0 - 0.5 * d * slope], np.full(d, slope)))
    if model in ("M3", "M4"):
        # Each 5 x^4 term projects to 4 x - 1.
        # The P2 products are orthogonal to the intercept and all covariates.
        return np.concatenate(([-1.0 * d], np.full(d, 4.0)))
    raise ValueError(f"unknown model {model!r}")


def generate(model, n, N, d, seed, noise_sigma=1.0):
    rng = np.random.default_rng(seed)
    X_all = rng.uniform(0.0, 1.0, size=(n + N, d))
    m = regression_function(model, X_all)
    eps = rng.normal(0.0, noise_sigma, size=n)
    return X_all[:n], m[:n] + eps, X_all[n:], m[n:]  # X, y, X_u and m(X_u)
