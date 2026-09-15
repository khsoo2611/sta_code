import numpy as np


def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def solve(A, b):
    """np.linalg.solve, falling back to least squares when A is singular."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def ols_beta(X, y):
    X1 = add_intercept(X)
    return solve(X1.T @ X1, X1.T @ y)


def sq_error(beta, beta0):
    return float(np.linalg.norm(np.reshape(beta, -1) - beta0) ** 2)


def kfold(n, k, seed):
    """Random k-fold split of range(n) as (train, validation) index pairs."""
    perm = np.random.default_rng(seed).permutation(n)
    return [(np.delete(np.arange(n), val), val) for val in np.array_split(perm, k)]


def n1_5_grid(n):
    """20 bandwidths c n^(-1/5), geometric in c from 0.35 up to h = 0.5."""
    base = n ** (-1 / 5)
    return base * np.geomspace(0.35, 0.5 / base, 20)


def theory_grid(n, d):
    """h_k = n^(-k / (21 d)), k = 1, ..., 20."""
    return n ** -(np.arange(1, 21) / (21 * d))


def optimal_grid(n, d):
    """20 bandwidths c n^(-1/(d+4)), c geometric in [0.05, 1] (EASE nuisance fits)."""
    return n ** (-1 / (d + 4)) * np.geomspace(0.05, 1.0, 20)
