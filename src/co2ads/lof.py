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


def scaled(df: pd.DataFrame, cols: list[str], scale: str = "minmax") -> np.ndarray:
    """Min–max (Module 1 normalisation) or z-scores, so neither column's units dominate the distance."""
    X = df[cols].to_numpy(float)
    if scale == "minmax":
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    if scale == "z":
        return (X - X.mean(axis=0)) / X.std(axis=0)
    return X


def lof_table(df: pd.DataFrame, cols: list[str], k: int, label: str = "country", scale: str = "minmax") -> pd.DataFrame:
    """LOF on scaled columns (min–max by default) so units don't dominate."""
    out = lof(scaled(df, cols, scale), k)
    out.insert(0, label, df[label].to_numpy())
    for c in cols:
        out[c] = df[c].to_numpy()
    return out.sort_values("lof", ascending=False).reset_index(drop=True)


def sklearn_check(X, k: int) -> dict:
    """Compare the hand-written LOF with sklearn LocalOutlierFactor(metric='manhattan') on the same points.

    sklearn always takes exactly k neighbours; the slides include every point tied at the k-distance.
    The two can only differ when such ties exist, so these are counted too.
    """
    from sklearn.neighbors import LocalOutlierFactor

    X = np.asarray(X, float)
    mine = lof(X, k)
    sk = -LocalOutlierFactor(n_neighbors=k, metric="manhattan").fit(X).negative_outlier_factor_
    diff = np.abs(mine["lof"].to_numpy() - sk)
    return {"k": k, "n": int(X.shape[0]), "max_abs_diff": float(diff.max()),
            "n_tied_neighbourhoods": int((mine["n_neighbours"] > k).sum()),
            "same_top10": bool(set(np.argsort(-mine["lof"].to_numpy())[:10]) == set(np.argsort(-sk)[:10]))}
