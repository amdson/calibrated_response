"""Benchmarks for the MaxEnt-SMM solver pipeline.

Measures time for each component of the solve loop:
  - JIT compilation (build + first iteration)
  - Steady-state per-iteration phase breakdown (HMC, features, grad-theta, update)
  - Scaling with n_vars, n_features, n_chains

Usage::

    python benchmarks/benchmark_solver.py
    python benchmarks/benchmark_solver.py --quick
    python benchmarks/benchmark_solver.py --save results.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from calibrated_response.maxent_smm.features import (
    MomentFeature,
    SoftThresholdFeature,
    compile_feature_vector,
)
from calibrated_response.maxent_smm.maxent_solver import JAXSolverConfig, MaxEntSolver
from calibrated_response.maxent_smm.mcmc import advance_buffer
from calibrated_response.maxent_smm.variable_spec import GaussianPriorSpec, VariableSpec


# ---------------------------------------------------------------------------
# Synthetic problem factory
# ---------------------------------------------------------------------------

def make_problem(
    n_vars: int,
    n_features: int,
    n_chains: int = 128,
    n_iterations: int = 1,
    seed: int = 42,
) -> tuple:
    """Build a synthetic MaxEnt-SMM problem.

    Returns (var_specs, feature_specs, feature_targets, energy_fn, init_theta, cfg).
    Feature mix: one mean constraint per variable, remainder are soft-threshold
    constraints at randomly sampled thresholds.
    """
    rng = np.random.default_rng(seed)

    var_specs = [
        VariableSpec(
            name=f"x{i}",
            description=f"Variable {i}",
            type="continuous",
            prior=GaussianPriorSpec(mean=0.5, std=0.25),
        )
        for i in range(n_vars)
    ]

    feature_specs: list = []
    for i in range(n_vars):
        feature_specs.append(MomentFeature(var_idx=i, order=1))
    while len(feature_specs) < n_features:
        k = len(feature_specs)
        var_idx = k % n_vars
        threshold = float(rng.uniform(0.2, 0.8))
        direction = "greater" if rng.random() > 0.5 else "less"
        feature_specs.append(
            SoftThresholdFeature(var_idx=var_idx, threshold=threshold, direction=direction)
        )
    feature_specs = feature_specs[:n_features]

    feature_targets = jnp.array(
        rng.uniform(0.2, 0.8, size=n_features).astype(np.float32)
    )

    # Linear energy: E_θ(x) = θ · f(x)
    fv_fn = compile_feature_vector(feature_specs)

    def energy_fn(theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(theta, fv_fn(x))

    init_theta = jnp.zeros(n_features, dtype=jnp.float32)

    cfg = JAXSolverConfig(
        num_chains=n_chains,
        num_iterations=n_iterations,
        verbose=False,
        adapt_step_size=True,
    )

    return var_specs, feature_specs, feature_targets, energy_fn, init_theta, cfg


def _build_solver(n_vars: int, n_features: int, n_chains: int, seed: int = 42) -> MaxEntSolver:
    """Construct and ``build()`` a solver (does not call ``solve()``)."""
    var_specs, feature_specs, targets, energy_fn, init_theta, cfg = make_problem(
        n_vars, n_features, n_chains, n_iterations=1, seed=seed
    )
    solver = MaxEntSolver(cfg)
    solver.build(
        var_specs=var_specs,
        feature_specs=feature_specs,
        feature_targets=targets,
        energy_fn=energy_fn,
        init_theta=init_theta,
    )
    return solver


def _warmup(solver: MaxEntSolver) -> None:
    """Run one iteration to trigger JAX JIT compilation of all kernels."""
    solver.solve()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BuildTimings:
    """Timings for the build and first-iteration (JIT compile) phases."""
    n_vars: int
    n_features: int
    n_chains: int
    build_sec: float        # solver.build() — Python setup, no JIT
    first_iter_sec: float   # first solver.solve() — triggers JAX tracing + XLA compile


@dataclass
class PhaseTimings:
    """Per-iteration timing breakdown for a single solve iteration."""
    n_vars: int
    n_features: int
    n_chains: int
    hmc_sec: float          # advance_buffer() — HMC proposals across all chains
    features_sec: float     # _batch_feature_fn() — feature computation
    grad_theta_sec: float   # _batch_grad_theta_fn() — energy gradient w.r.t. θ
    update_sec: float       # gradient assembly + optax parameter update
    total_sec: float        # wall time for the full iteration


# ---------------------------------------------------------------------------
# Build-time benchmark
# ---------------------------------------------------------------------------

def bench_build_time(n_vars: int, n_features: int, n_chains: int) -> BuildTimings:
    """Measure ``build()`` and first-iteration (JIT compile) times."""
    var_specs, feature_specs, targets, energy_fn, init_theta, cfg = make_problem(
        n_vars, n_features, n_chains, n_iterations=1
    )
    solver = MaxEntSolver(cfg)

    t0 = time.perf_counter()
    solver.build(
        var_specs=var_specs,
        feature_specs=feature_specs,
        feature_targets=targets,
        energy_fn=energy_fn,
        init_theta=init_theta,
    )
    build_sec = time.perf_counter() - t0

    # First call to solve() traces JAX programs and compiles XLA kernels.
    t0 = time.perf_counter()
    solver.solve()
    first_iter_sec = time.perf_counter() - t0

    return BuildTimings(
        n_vars=n_vars,
        n_features=n_features,
        n_chains=n_chains,
        build_sec=build_sec,
        first_iter_sec=first_iter_sec,
    )


# ---------------------------------------------------------------------------
# Steady-state iteration phase breakdown
# ---------------------------------------------------------------------------

def bench_iteration_phases(
    n_vars: int,
    n_features: int,
    n_chains: int,
    n_repeats: int = 20,
) -> list[PhaseTimings]:
    """Measure per-phase timings for steady-state solver iterations.

    Phases timed independently:
      1. HMC ``advance_buffer()``
      2. Batch feature computation ``_batch_feature_fn()``
      3. Batch grad-theta computation ``_batch_grad_theta_fn()``
      4. Gradient assembly + optax parameter update

    JAX kernels are warmed up (JIT-compiled) before timing begins.
    """
    solver = _build_solver(n_vars, n_features, n_chains)
    _warmup(solver)  # JIT compile all kernels

    cfg = solver.config
    theta = solver._theta
    buffer = solver._buffer
    opt_state = solver._opt_state
    optimizer = solver._optimizer
    hmc_cfg = solver._hmc_config
    targets = solver._feature_targets

    results: list[PhaseTimings] = []

    for _ in range(n_repeats):
        t_start = time.perf_counter()

        # Phase 1: HMC advance
        t0 = time.perf_counter()
        buffer = advance_buffer(
            buffer=buffer,
            theta=theta,
            n_steps=cfg.mcmc_steps_per_iteration,
            hmc_config=hmc_cfg,
            step_fn=solver._hmc_step_fn,
        )
        jax.block_until_ready(buffer.states)
        hmc_sec = time.perf_counter() - t0

        # Phase 2: batch feature computation
        t0 = time.perf_counter()
        chain_features = solver._batch_feature_fn(buffer.states)
        jax.block_until_ready(chain_features)
        features_sec = time.perf_counter() - t0

        # Phase 3: batch gradient of energy w.r.t. θ
        t0 = time.perf_counter()
        chain_grad_theta = solver._batch_grad_theta_fn(theta, buffer.states)
        jax.block_until_ready(chain_grad_theta)
        grad_theta_sec = time.perf_counter() - t0

        # Phase 4: gradient assembly + optax update
        t0 = time.perf_counter()
        model_expectations = chain_features.mean(axis=0)
        delta = model_expectations - targets
        centered = chain_features - model_expectations
        w = -(centered @ delta)
        grad = (w[:, None] * chain_grad_theta).mean(axis=0)
        grad = grad + cfg.l2_regularization * theta
        grad = jnp.clip(grad, -cfg.grad_clip, cfg.grad_clip)
        updates, opt_state = optimizer.update(grad, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        jax.block_until_ready(theta)
        update_sec = time.perf_counter() - t0

        total_sec = time.perf_counter() - t_start
        results.append(
            PhaseTimings(
                n_vars=n_vars,
                n_features=n_features,
                n_chains=n_chains,
                hmc_sec=hmc_sec,
                features_sec=features_sec,
                grad_theta_sec=grad_theta_sec,
                update_sec=update_sec,
                total_sec=total_sec,
            )
        )

    return results


def _summarise_phases(timings: list[PhaseTimings]) -> dict:
    """Return median + std for each phase across repeated measurements."""
    def _arr(attr):
        return np.array([getattr(t, attr) for t in timings])

    out = {}
    for field in ("hmc_sec", "features_sec", "grad_theta_sec", "update_sec", "total_sec"):
        arr = _arr(field)
        out[field] = float(np.median(arr))
        out[f"{field}_std"] = float(np.std(arr))
    return out


# ---------------------------------------------------------------------------
# Full solve timing
# ---------------------------------------------------------------------------

def bench_full_solve(
    n_vars: int,
    n_features: int,
    n_chains: int,
    n_iterations: int = 100,
) -> dict:
    """Run a complete ``solve()`` and return per-iteration timing history."""
    var_specs, feature_specs, targets, energy_fn, init_theta, _ = make_problem(
        n_vars, n_features, n_chains, n_iterations=n_iterations
    )
    cfg = JAXSolverConfig(
        num_chains=n_chains,
        num_iterations=n_iterations,
        verbose=False,
    )
    solver = MaxEntSolver(cfg)
    solver.build(
        var_specs=var_specs,
        feature_specs=feature_specs,
        feature_targets=targets,
        energy_fn=energy_fn,
        init_theta=init_theta,
    )

    t0 = time.perf_counter()
    _, info = solver.solve()
    total_sec = time.perf_counter() - t0

    history = info["history"]
    per_iter = history["runtime_sec"]

    return {
        "total_sec": total_sec,
        "n_iterations": info["n_iterations"],
        "converged": info["converged"],
        "per_iter_sec": per_iter,
        "mean_iter_sec": float(np.mean(per_iter)),
        "median_iter_sec": float(np.median(per_iter)),
        # Separate compile iteration (first) from steady state (rest)
        "compile_iter_sec": per_iter[0] if per_iter else None,
        "steady_state_mean_sec": float(np.mean(per_iter[1:])) if len(per_iter) > 1 else None,
        "config": {
            "n_vars": n_vars,
            "n_features": n_features,
            "n_chains": n_chains,
            "n_iterations": n_iterations,
        },
    }


# ---------------------------------------------------------------------------
# Scaling sweeps
# ---------------------------------------------------------------------------

def sweep_n_vars(
    n_vars_list: list[int],
    n_features_base: int = 8,
    n_chains: int = 128,
    n_repeats: int = 10,
) -> list[dict]:
    """Measure iteration time as the number of variables increases."""
    results = []
    for n_vars in n_vars_list:
        n_features = max(n_features_base, n_vars)
        print(f"  n_vars={n_vars}, n_features={n_features} ...", flush=True)
        timings = bench_iteration_phases(n_vars, n_features, n_chains, n_repeats)
        row = {"n_vars": n_vars, "n_features": n_features}
        row.update(_summarise_phases(timings))
        results.append(row)
    return results


def sweep_n_features(
    n_features_list: list[int],
    n_vars: int = 3,
    n_chains: int = 128,
    n_repeats: int = 10,
) -> list[dict]:
    """Measure iteration time as the number of features (constraints) increases."""
    results = []
    for n_features in n_features_list:
        print(f"  n_features={n_features} ...", flush=True)
        timings = bench_iteration_phases(n_vars, n_features, n_chains, n_repeats)
        row = {"n_vars": n_vars, "n_features": n_features}
        row.update(_summarise_phases(timings))
        results.append(row)
    return results


def sweep_n_chains(
    n_chains_list: list[int],
    n_vars: int = 3,
    n_features: int = 8,
    n_repeats: int = 10,
) -> list[dict]:
    """Measure iteration time as the number of parallel HMC chains increases."""
    results = []
    for n_chains in n_chains_list:
        print(f"  n_chains={n_chains} ...", flush=True)
        timings = bench_iteration_phases(n_vars, n_features, n_chains, n_repeats)
        row = {"n_chains": n_chains}
        row.update(_summarise_phases(timings))
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

_QUICK = dict(
    n_repeats=5,
    n_iterations=30,
    n_vars_list=[2, 3, 5],
    n_features_list=[4, 8, 16],
    n_chains_list=[32, 64, 128],
)

_FULL = dict(
    n_repeats=20,
    n_iterations=100,
    n_vars_list=[2, 3, 4, 5, 6, 8],
    n_features_list=[4, 8, 12, 16, 24, 32],
    n_chains_list=[32, 64, 128, 256],
)

_DEFAULTS = dict(n_vars=3, n_features=10, n_chains=128)


def run_benchmarks(quick: bool = False) -> dict:
    """Run the full benchmark suite and return a nested results dict."""
    preset = _QUICK if quick else _FULL
    dv = _DEFAULTS

    results: dict = {}

    print("[1/4] Build & JIT compile times ...", flush=True)
    results["build_times"] = asdict(bench_build_time(**dv))

    print("[2/4] Steady-state iteration phase breakdown ...", flush=True)
    phase_timings = bench_iteration_phases(
        n_vars=dv["n_vars"],
        n_features=dv["n_features"],
        n_chains=dv["n_chains"],
        n_repeats=preset["n_repeats"],
    )
    results["phase_breakdown"] = {
        "summary": _summarise_phases(phase_timings),
        "all_iterations": [asdict(t) for t in phase_timings],
        "config": dv,
    }

    print("[3/4] Scaling sweeps ...", flush=True)
    print("  Sweeping n_vars ...")
    results["sweep_n_vars"] = sweep_n_vars(
        n_vars_list=preset["n_vars_list"],
        n_features_base=dv["n_features"],
        n_chains=dv["n_chains"],
        n_repeats=preset["n_repeats"],
    )
    print("  Sweeping n_features ...")
    results["sweep_n_features"] = sweep_n_features(
        n_features_list=preset["n_features_list"],
        n_vars=dv["n_vars"],
        n_chains=dv["n_chains"],
        n_repeats=preset["n_repeats"],
    )
    print("  Sweeping n_chains ...")
    results["sweep_n_chains"] = sweep_n_chains(
        n_chains_list=preset["n_chains_list"],
        n_vars=dv["n_vars"],
        n_features=dv["n_features"],
        n_repeats=preset["n_repeats"],
    )

    print("[4/4] Full solve timing ...", flush=True)
    results["full_solve"] = bench_full_solve(
        n_vars=dv["n_vars"],
        n_features=dv["n_features"],
        n_chains=dv["n_chains"],
        n_iterations=preset["n_iterations"],
    )

    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _ms(sec: Optional[float]) -> str:
    if sec is None:
        return "  N/A"
    return f"{sec * 1000:8.2f} ms"


def print_results(results: dict) -> None:
    """Print a human-readable summary of benchmark results to stdout."""
    bar = "=" * 62
    print(f"\n{bar}")
    print("  MaxEnt-SMM Solver Benchmark Results")
    print(bar)

    bt = results.get("build_times", {})
    print("\n[Build & JIT Compile]")
    print(f"  build()            : {_ms(bt.get('build_sec'))}")
    print(f"  first iteration    : {_ms(bt.get('first_iter_sec'))}  (JAX trace + XLA compile)")

    pb = results.get("phase_breakdown", {})
    if pb:
        s = pb["summary"]
        total = s["total_sec"]
        cfg = pb.get("config", {})
        print(f"\n[Per-Iteration Phase Breakdown]  config={cfg}")
        print(f"  {'Phase':<22}  {'Median':>10}  {'% of iter':>10}")
        print(f"  {'-'*22}  {'-'*10}  {'-'*10}")
        for label, key in [
            ("HMC advance_buffer", "hmc_sec"),
            ("Batch features", "features_sec"),
            ("Batch grad-θ", "grad_theta_sec"),
            ("Optax update", "update_sec"),
        ]:
            val = s[key]
            pct = 100 * val / total if total > 0 else 0
            print(f"  {label:<22}  {_ms(val)}  {pct:>9.1f}%")
        print(f"  {'Total':<22}  {_ms(total)}")

    fs = results.get("full_solve", {})
    if fs:
        cfg = fs.get("config", {})
        print(f"\n[Full Solve]  config={cfg}")
        print(f"  Total wall time    : {fs['total_sec']:.3f} s")
        print(f"  Iterations run     : {fs['n_iterations']}")
        print(f"  Converged          : {fs['converged']}")
        print(f"  First iter (compile+run) : {_ms(fs.get('compile_iter_sec'))}")
        print(f"  Steady-state mean  : {_ms(fs.get('steady_state_mean_sec'))}")
        print(f"  Median / iter      : {_ms(fs.get('median_iter_sec'))}")

    for label, key, param in [
        ("n_vars", "sweep_n_vars", "n_vars"),
        ("n_features", "sweep_n_features", "n_features"),
        ("n_chains", "sweep_n_chains", "n_chains"),
    ]:
        data = results.get(key, [])
        if not data:
            continue
        print(f"\n[Scaling: {label}]")
        header = (
            f"  {label:>10}  {'total':>10}  {'HMC':>10}  {'features':>10}  {'grad-θ':>10}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in data:
            print(
                f"  {row[param]:>10}  "
                f"{_ms(row['total_sec'])}  "
                f"{_ms(row['hmc_sec'])}  "
                f"{_ms(row['features_sec'])}  "
                f"{_ms(row['grad_theta_sec'])}"
            )

    print(f"\n{bar}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the MaxEnt-SMM solver pipeline components."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shorter runs (fewer repeats, smaller sweep ranges).",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save JSON results to this file path.",
    )
    args = parser.parse_args()

    print(f"JAX backend : {jax.default_backend()}")
    print(f"JAX devices : {jax.devices()}\n")

    results = run_benchmarks(quick=args.quick)
    print_results(results)

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
