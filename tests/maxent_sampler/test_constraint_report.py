"""constraint_report hinge handling: satisfied one-sided constraints must
report error_rel == 0.0 (only same-side deviations count), while the raw
signed slack stays in `error`. Before the Aug-2026 fix a satisfied
`E[x] > 100` with fitted 5000 topped every "worst constraints" list."""
import numpy as np
import pytest

from calibrated_response.models.natural_response import parse_natural_syntax
from calibrated_response.models.variable import (BinaryVariable,
                                                 ContinuousVariable)
from calibrated_response.maxent_sampler.distribution_builder import (
    DistributionBuilder)

VARS = [
    ContinuousVariable(name="x", description="d",
                       lower_bound=0.0, upper_bound=100.0),
    ContinuousVariable(name="y", description="d",
                       lower_bound=0.0, upper_bound=100.0),
    BinaryVariable(name="b", description="d"),
]

EXPRS = [
    "E[x] = 70 ~ 5",                 # point anchor pulls x high
    "E[x] > 5",                      # satisfied floor, huge slack
    "P(x > 95) < 0.5",               # satisfied ceiling, big slack
    "Corr(x, y) > -0.9",             # satisfied corr floor
    "E[y] < 10 ~ 30",                # weak ceiling the fit should violate...
    "E[y] = 60 ~ 2",                 # ...because the strong anchor wins y=60
    "y < x ~ N(0, 5)",               # equation inequality (one-sided ev)
    "P(b = True) = 0.3",
]


@pytest.fixture(scope="module")
def report():
    # a LIST, not an id-keyed dict: parse-generated ids are not unique
    # ('E[x] = 70' and 'E[x] > 5' both id as 'E_x_')
    b = DistributionBuilder(VARS, [parse_natural_syntax(e) for e in EXPRS])
    assert not b.skipped, b.skipped
    b.fit(steps=300, lr=2e-3, n_samples=256, seed=0)
    return b.constraint_report(n_samples=20_000)


def _row(report, frag):
    hits = [r for r in report if frag in r["estimate"]]
    assert len(hits) == 1, (frag, [r["estimate"] for r in report])
    return hits[0]


def test_rows_carry_relation(report):
    rels = {r["estimate"]: r["relation"] for r in report}
    assert rels["E[x] > 5.0"] == "ge"
    assert rels["P(x > 95.0) < 0.5"] == "le"
    assert rels["E[x] = 70.0"] == "eq"


def test_satisfied_hinges_report_zero_violation(report):
    for frag in ("E[x] > 5.0", "P(x > 95.0) < 0.5", "Corr(x, y) > -0.9"):
        r = _row(report, frag)
        assert r["error_rel"] == 0.0, (frag, r)
        # the raw slack is preserved for reading
        assert r["error"] != 0.0


def test_violated_hinge_reports_violation(report):
    # E[y] < 10 while an eq anchor holds y near 60: same-side deviation
    r = _row(report, "E[y] < 10.0")
    assert r["fitted"] > 15.0, r
    assert r["error_rel"] > 0.0, r


def test_eq_rows_keep_signed_residual(report):
    r = _row(report, "E[x] = 70.0")
    assert r["error_rel"] == pytest.approx(r["error"] / 100.0)


def test_equation_inequality_not_double_hinged(report):
    # ev already returns one-sided RMS violation (>= 0); the report must not
    # fold it again (relation stays "eq" on equation rows by design)
    r = _row(report, "y < x")
    assert r["relation"] == "eq"
    assert r["error_rel"] >= 0.0


def test_sorting_surfaces_the_real_misfit(report):
    # among the P/E/Corr rows the violated ceiling must rank worst — the
    # satisfied hinges all sit at 0.0 (equation rows report raw-unit RMS on
    # their own scale, so they are excluded from this comparison)
    rows = [r for r in report if not r["id"].startswith("EQ_")]
    worst = max(rows, key=lambda r: abs(r["error_rel"]))
    assert "E[y] < 10.0" == worst["estimate"], worst
