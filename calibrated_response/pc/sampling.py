"""Ancestral sampling from a trained circuit (spec §9).

Implemented in numpy on host: sampling has data-dependent control flow per node
and is only needed for validation (spec §8 test 4) and downstream generative use,
so it need not be differentiable or jit-compiled. At each sum node we pick a
child component by its weights, recurse, and sample leaves.
"""

from __future__ import annotations

import jax
import numpy as np


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def sample(circuit, params, n_samples, seed=0):
    """Draw ``n_samples`` from the circuit. Returns ``(n_samples, n_vars)`` array
    in original variable domains; column order matches ``circuit.var_specs``."""
    # Drop-in for the Gaussian baseline, which samples from its own joint Gaussian.
    if getattr(circuit, "is_gaussian_baseline", False):
        return circuit.sample(params, n_samples, seed)

    rng = np.random.default_rng(seed)
    rg = circuit.rg
    C = circuit.C

    np_params = jax.tree_util.tree_map(np.asarray, params)
    layer_W = np_params.get("layer_W", [])
    root_W = np.asarray(np_params["root_W"])

    g_r2l = {int(r): i for i, r in enumerate(circuit.g_region)}
    c_r2l = {int(r): i for i, r in enumerate(circuit.c_region)}

    def descend(region, comp, out):
        info = rg.region_info[region]
        if info[0] == "internal":
            _, left, right, layer_idx, local = info
            w = layer_W[layer_idx][local, comp]               # (C, C) logits over (i, j)
            flat = _softmax(w.reshape(-1))
            idx = rng.choice(C * C, p=flat)
            i, j = idx // C, idx % C
            descend(left, i, out)
            descend(right, j, out)
            return
        var = info[1]
        if region in g_r2l:
            l = g_r2l[region]
            k = rng.choice(circuit.K, p=_softmax(np_params["g_mix"][l, comp]))
            mu = np_params["g_mu"][l, k]
            sigma = np.exp(0.5 * np_params["g_logvar"][l, k])
            val = rng.normal(mu, sigma)
            val = np.clip(val, circuit.lower[var], circuit.upper[var])
            out[var] = val
        else:
            l = c_r2l[region]
            k = rng.choice(circuit.K, p=_softmax(np_params["c_mix"][l, comp]))
            logits = np_params["c_logits"][l, k] + np.log(np.asarray(circuit.c_valid)[l] + 1e-30)
            m = rng.choice(circuit.M, p=_softmax(logits))
            out[var] = np.asarray(circuit.c_values)[l, m]

    root_p = _softmax(root_W)
    samples = np.zeros((n_samples, circuit.n_vars), dtype=np.float64)
    for s in range(n_samples):
        idx = rng.choice(circuit.R * C, p=root_p)
        rep, comp = idx // C, idx % C
        descend(int(rg.rep_root_id[rep]), comp, samples[s])
    return samples
