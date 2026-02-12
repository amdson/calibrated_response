from __future__ import annotations

import numpy as np

from calibrated_response.maxent_pgmax.solver import PGMaxMaxEntSolver, SolverConfig
from calibrated_response.models.query import (
    ConditionalProbabilityEstimate,
    EqualityProposition,
    ExpectationEstimate,
    InequalityProposition,
    ProbabilityEstimate,
)
from calibrated_response.models.variable import BinaryVariable, ContinuousVariable


def _variables() -> list[ContinuousVariable | BinaryVariable]:
    return [
        ContinuousVariable(
            name="x",
            description="Target",
            lower_bound=0.0,
            upper_bound=1.0,
            unit="u",
        ),
        BinaryVariable(name="b", description="Condition flag"),
    ]


def test_solver_reports_skipped_for_unsupported_multi_condition(monkeypatch) -> None:
    solver = PGMaxMaxEntSolver(_variables(), config=SolverConfig(max_bins=8))

    def fake_run_bp(
        unary_evidence: dict[str, np.ndarray],
        pairwise_potentials: dict[tuple[str, str], np.ndarray],
    ) -> list[np.ndarray]:
        return [
            np.ones_like(unary_evidence["x"]) / unary_evidence["x"].size,
            np.ones_like(unary_evidence["b"]) / unary_evidence["b"].size,
        ]

    monkeypatch.setattr(solver, "_run_bp", fake_run_bp)

    estimate = ConditionalProbabilityEstimate(
        id="cp1",
        proposition=InequalityProposition(
            variable="x",
            variable_type="continuous",
            threshold=0.7,
            is_lower_bound=True,
        ),
        conditions=[
            EqualityProposition(variable="b", variable_type="binary", value=True),
            InequalityProposition(
                variable="x",
                variable_type="continuous",
                threshold=0.3,
                is_lower_bound=True,
            ),
        ],
        probability=0.6,
    )

    _, _, info = solver.solve([estimate])

    assert any("only one-condition" in msg for msg in info["skipped_constraints"])


def test_probability_and_expectation_estimates_tilt_unary_evidence(monkeypatch) -> None:
    solver = PGMaxMaxEntSolver(_variables(), config=SolverConfig(max_bins=12))

    captured: dict[str, np.ndarray] = {}

    def fake_run_bp(
        unary_evidence: dict[str, np.ndarray],
        pairwise_potentials: dict[tuple[str, str], np.ndarray],
    ) -> list[np.ndarray]:
        captured.update({k: v.copy() for k, v in unary_evidence.items()})
        return [
            np.ones_like(unary_evidence["x"]) / unary_evidence["x"].size,
            np.ones_like(unary_evidence["b"]) / unary_evidence["b"].size,
        ]

    monkeypatch.setattr(solver, "_run_bp", fake_run_bp)

    estimates = [
        ProbabilityEstimate(
            id="p1",
            proposition=InequalityProposition(
                variable="x",
                variable_type="continuous",
                threshold=0.6,
                is_lower_bound=True,
            ),
            probability=0.8,
        ),
        ExpectationEstimate(id="e1", variable="x", expected_value=0.25),
    ]

    solver.solve(estimates)

    assert "x" in captured
    assert np.any(np.abs(captured["x"]) > 0.0)
