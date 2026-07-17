"""Dataset loaders. Each returns (train_df, test_df) of analysis-ready columns.

Real tables come from OpenML / sklearn (cached locally after first fetch) and are
chosen to span table types: ``adult`` and ``bank_marketing`` are mixed,
``california`` is all-continuous, ``nursery`` is all-categorical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# fnlwgt is a survey weight, education-num duplicates education
_ADULT_DROP = ["fnlwgt", "education-num"]


def _split(df: pd.DataFrame, test_frac: float, seed: int,
           max_rows: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    if max_rows is not None:
        perm = perm[:max_rows]
    n_test = int(len(perm) * test_frac)
    return df.iloc[perm[n_test:]].reset_index(drop=True), \
           df.iloc[perm[:n_test]].reset_index(drop=True)


def _drop_rare_levels(df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    """Drop rows holding a categorical level rarer than ``min_count``.

    ``TableEncoder.bin_indices`` raises on a test-split level unseen in train;
    culling ultra-rare levels before the random split makes that (near-)impossible.
    """
    mask = pd.Series(True, index=df.index)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            counts = df[col].astype(str).value_counts()
            keep = counts[counts >= min_count].index
            mask &= df[col].astype(str).isin(keep)
    return df[mask]


def load_adult(test_frac: float = 0.2, seed: int = 0,
               max_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """UCI Adult via OpenML (cached by sklearn after first fetch). 13 columns, mixed."""
    from sklearn.datasets import fetch_openml

    df = fetch_openml("adult", version=2, as_frame=True).frame
    df = _drop_rare_levels(df.drop(columns=_ADULT_DROP).dropna())
    return _split(df, test_frac, seed, max_rows)


# OpenML data_id 1461 ships the UCI bank-marketing table with anonymized V1..V16
# names; restore the documented UCI column names.
_BANK_COLS = ["age", "job", "marital", "education", "default", "balance",
              "housing", "loan", "contact", "day", "month", "duration",
              "campaign", "pdays", "previous", "poutcome", "subscribed"]


def load_bank_marketing(test_frac: float = 0.2, seed: int = 0,
                        max_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """UCI Bank Marketing via OpenML. ~45k rows, 17 columns, categorical-heavy."""
    from sklearn.datasets import fetch_openml

    df = fetch_openml(data_id=1461, as_frame=True).frame
    df.columns = _BANK_COLS
    df = _drop_rare_levels(df.dropna())
    return _split(df, test_frac, seed, max_rows)


def load_california(test_frac: float = 0.2, seed: int = 0,
                    max_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """California housing (sklearn). ~21k rows, 9 columns, all continuous."""
    from sklearn.datasets import fetch_california_housing

    df = fetch_california_housing(as_frame=True).frame
    return _split(df.dropna(), test_frac, seed, max_rows)


def load_nursery(test_frac: float = 0.2, seed: int = 0,
                 max_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """UCI Nursery via OpenML. ~13k rows, 9 columns, all categorical."""
    from sklearn.datasets import fetch_openml

    df = fetch_openml("nursery", version=1, as_frame=True).frame
    df = _drop_rare_levels(df.dropna())
    return _split(df, test_frac, seed, max_rows)


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
    "bank_marketing": load_bank_marketing,
    "california": load_california,
    "nursery": load_nursery,
    "synthetic_chain": load_synthetic_chain,
}
