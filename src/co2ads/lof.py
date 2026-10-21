"""Module 5 — Local Outlier Factor with Manhattan distance, step by step as on the slides.

    1a  dist(p, q) = |x1 − x2| + |y1 − y2| + ...
    1b  k-distance(p) = distance to the k-th nearest neighbour; N_k(p) = points within it (ties included)
    2   reach-dist(p, o) = max(k-distance(o), dist(p, o))
    3   lrd(p) = |N_k(p)| / Σ reach-dist(p, o)
    4   LOF(p) = mean lrd(o) over N_k(p) / lrd(p)       (≈1 normal, ≫1 outlier)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def manhattan_matrix(X) -> np.ndarray:
    X = np.asarray(X, float)
    return np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)


def lof(X, k: int) -> pd.DataFrame:
    X = np.asarray(X, float)
    n = X.shape[0]
    if not 1 <= k < n:
        raise ValueError("k must be between 1 and n − 1")
    d = manhattan_matrix(X)
    np.fill_diagonal(d, np.inf)                       # a point is not its own neighbour
    kdist = np.sort(d, axis=1)[:, k - 1]
    neigh = [np.flatnonzero(d[p] <= kdist[p]) for p in range(n)]
    lrd = np.empty(n)
    for p in range(n):
        o = neigh[p]
        reach = np.maximum(kdist[o], d[p, o])
        lrd[p] = len(o) / reach.sum()
    score = np.array([lrd[neigh[p]].mean() / lrd[p] for p in range(n)])
    return pd.DataFrame({"k_distance": kdist, "n_neighbours": [len(o) for o in neigh],
                         "lrd": lrd, "lof": score})


def lof_table(df: pd.DataFrame, cols: list[str], k: int, label: str = "country", scale: str = "minmax") -> pd.DataFrame:
    """LOF on scaled columns (min–max by default, Module 1 normalisation) so units don't dominate."""
    X = df[cols].to_numpy(float)
    if scale == "minmax":
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    elif scale == "z":
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    out = lof(X, k)
    out.insert(0, label, df[label].to_numpy())
    for c in cols:
        out[c] = df[c].to_numpy()
    return out.sort_values("lof", ascending=False).reset_index(drop=True)
