"""Mixed-type table -> shared discrete grid.

Every engine in the benchmark sees the *same* encoding, so scores are
comparable: each column becomes an integer site.

- Continuous columns are binned on **train-split quantiles** (so uniform bins in
  encoded space = quantile bins in raw space; heavy tails and spikes like
  ``capital-gain`` collapse duplicate quantile edges into fewer bins instead of
  wasting resolution).
- Categorical columns get one bin per observed level.

Engines that want a continuous embedding use the encoded value: bin ``b`` of a
``d``-bin site sits at ``(b + 0.5) / d`` in ``[0, 1]``. This matches
``ContinuousVar(name, 0.0, 1.0, n_bins=d)`` for the tensor chain, so
``TensorChain.disc.bin_centers`` and expectations line up with
:meth:`TableEncoder.centers` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from calibrated_response.tn.discretize import ContinuousVar


def _cont_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges; spike-heavy columns (e.g. capital-loss, ~95% zeros)
    collapse most quantiles onto the modal value, so when that happens the mode
    gets its own bin and the remaining values are quantile-binned separately."""
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) - 1 >= max(2, n_bins // 2):
        return edges
    vals, counts = np.unique(x, return_counts=True)
    mode = vals[counts.argmax()]
    rest = x[x != mode]
    if rest.size == 0:
        raise ValueError("column is constant")
    sub = np.quantile(rest, np.linspace(0.0, 1.0, n_bins))
    edges = np.unique(np.concatenate(
        [[x.min()], [np.nextafter(mode, np.inf)], sub, [x.max()]]))
    if len(edges) < 3:
        raise ValueError("column is (near-)constant")
    return edges


@dataclass(frozen=True)
class VarSpec:
    name: str
    kind: str                      # "cont" | "cat"
    n_bins: int
    levels: tuple = ()             # cat: level -> bin index is position here
    edges: tuple = ()              # cont: raw-space bin edges, len n_bins + 1


class TableEncoder:
    """Fit on the train split, then map any dataframe to integer bin indices."""

    def __init__(self, df: pd.DataFrame, n_bins_cont: int = 16,
                 cat_cols: list[str] | None = None):
        self.specs: list[VarSpec] = []
        for col in df.columns:
            is_cat = (cat_cols is not None and col in cat_cols) or \
                     not pd.api.types.is_numeric_dtype(df[col])
            if is_cat:
                levels = tuple(sorted(df[col].astype(str).unique()))
                self.specs.append(VarSpec(col, "cat", len(levels), levels=levels))
            else:
                x = df[col].to_numpy(dtype=np.float64)
                try:
                    edges = _cont_edges(x, n_bins_cont)
                except ValueError as e:
                    raise ValueError(f"column {col!r}: {e}") from None
                self.specs.append(VarSpec(col, "cont", len(edges) - 1,
                                          edges=tuple(edges)))
        self.names = [s.name for s in self.specs]
        self.dims = [s.n_bins for s in self.specs]
        self.site = {name: i for i, name in enumerate(self.names)}

    # ------------------------------------------------------------------
    def bin_indices(self, df: pd.DataFrame) -> np.ndarray:
        """(N, D) integer bin indices; unseen categorical levels raise."""
        cols = []
        for s in self.specs:
            if s.kind == "cat":
                lut = {lv: i for i, lv in enumerate(s.levels)}
                cols.append(df[s.name].astype(str).map(lut).to_numpy())
                if np.any(pd.isna(cols[-1])):
                    raise ValueError(f"unseen level in column {s.name!r}")
            else:
                x = df[s.name].to_numpy(dtype=np.float64)
                idx = np.searchsorted(np.asarray(s.edges)[1:-1], x, side="right")
                cols.append(np.clip(idx, 0, s.n_bins - 1))
        return np.stack(cols, axis=1).astype(np.int64)

    def centers(self, var: str) -> np.ndarray:
        """Encoded [0,1] bin centres of ``var`` — the value scale for expectations."""
        d = self.dims[self.site[var]]
        return (np.arange(d) + 0.5) / d

    def tn_vars(self) -> list[ContinuousVar]:
        return [ContinuousVar(s.name, 0.0, 1.0, n_bins=s.n_bins) for s in self.specs]
