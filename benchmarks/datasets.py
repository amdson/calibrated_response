"""Dataset loaders. Each returns (train_df, test_df) of analysis-ready columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

# fnlwgt is a survey weight, education-num duplicates education
_ADULT_DROP = ["fnlwgt", "education-num"]


def load_adult(test_frac: float = 0.2, seed: int = 0,
               max_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """UCI Adult via OpenML (cached by sklearn after first fetch). 13 columns."""
    from sklearn.datasets import fetch_openml

    df = fetch_openml("adult", version=2, as_frame=True).frame
    df = df.drop(columns=_ADULT_DROP).dropna()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    if max_rows is not None:
        perm = perm[:max_rows]
    n_test = int(len(perm) * test_frac)
    return df.iloc[perm[n_test:]].reset_index(drop=True), \
           df.iloc[perm[:n_test]].reset_index(drop=True)


def load_synthetic_chain(n_vars: int = 6, n_rows: int = 20_000, rho: float = 0.7,
                         seed: int = 0, test_frac: float = 0.2,
                         max_rows: int | None = None):
    """Gaussian AR(1) chain — a fully-known-truth debugging dataset."""
    if max_rows is not None:
        n_rows = max_rows
    rng = np.random.default_rng(seed)
    X = np.zeros((n_rows, n_vars))
    X[:, 0] = rng.normal(size=n_rows)
    for i in range(1, n_vars):
        X[:, i] = rho * X[:, i - 1] + np.sqrt(1 - rho ** 2) * rng.normal(size=n_rows)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_vars)])
    n_test = int(n_rows * test_frac)
    return df.iloc[n_test:].reset_index(drop=True), \
           df.iloc[:n_test].reset_index(drop=True)


DATASETS = {
    "adult": load_adult,
    "synthetic_chain": load_synthetic_chain,
}
