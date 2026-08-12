"""Compile equation strings into sample-native residual functions.

An ``EquationEstimate`` such as ``anomaly = trend + 0.075*enso_oni - volcanic``
(optionally ``~ N(0, sigma)``) is a statement about how a *residual* is
distributed.  Move everything to one side::

    r(x) = lhs - rhs                       # evaluated per sample, shape (N,)

and the equation becomes a claim about ``r``:

    deterministic   lhs = rhs              ->  r == 0        (identity)
    gaussian noise  lhs = rhs + N(0, s)    ->  r ~ N(0, s)   (mean 0, spread s)
    inequality      lhs > rhs / lhs < rhs  ->  r >= 0 / r <= 0

All three are cheap sample losses (see ``maxent_sampler.model``): the
deterministic form drives ``mean(r**2) -> 0``; the noisy form matches
``mean(r) -> 0`` and ``mean(r**2) -> s**2``.  Maxent supplies the rest — given
only those two moments of the residual, the maximum-entropy completion makes
``r`` Gaussian and approximately independent of the right-hand-side variables,
i.e. the structural equation ``lhs = rhs + independent noise`` — without ever
encoding independence.

The inequality form is different in kind: ``mean(relu(-r)**2) -> 0`` restricts
the *support* to one side of the surface ``lhs = rhs`` instead of pinning a
moment, so maxent spreads over the whole admissible region rather than hugging a
line.  Its optional ``N(0, s)`` sets how softly the boundary is enforced.

Which of the three applies is carried by the estimate (``relation`` +
``noise_sd``); this module only turns the *string* into two closures over the
sample matrix:

    soft, hard = compile_residual("anomaly", "trend + 0.075*enso_oni", idx, span, k)
    soft(x)   # jax, differentiable — used inside the loss (soft indicators)
    hard(x)   # numpy, exact indicators — used for readout / constraint_report

The grammar is arithmetic (``+ - * / **``, unary ``-``), numeric constants,
variable names, ``abs(e)``, ``min(a, b)``, ``max(a, b)`` and a threshold
indicator ``ind(expr > c)`` (``>`` ``>=`` ``<`` ``<=``), which is a smooth
sigmoid inside the loss and an exact step on readout.  Parsing goes through
Python's ``ast`` under a strict node whitelist — no attribute access, no
arbitrary calls — so an equation string cannot execute anything.
"""

from __future__ import annotations

import ast
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def _compare_scale(cmp: ast.Compare, span: dict) -> float:
    """Span of the variable in a ``var <> const`` comparison (else 1.0).

    Lets ``ind(anomaly > 1.60)`` use the same span-normalised sharpness the
    proposition indicators use; a variable-vs-variable comparison has no single
    natural scale, so it falls back to 1.0 (raw units)."""
    left, right = cmp.left, cmp.comparators[0]
    for a, b in ((left, right), (right, left)):
        if isinstance(a, ast.Name) and isinstance(b, ast.Constant) \
                and a.id in span:
            return float(span[a.id])
    return 1.0


def _compile_compare(cmp: ast.Compare, ctx: dict, xp, soft: bool) -> Callable:
    if len(cmp.ops) != 1:
        raise ValueError("ind() takes a single comparison, e.g. ind(x > 1.5)")
    op = cmp.ops[0]
    left = _compile(cmp.left, ctx, xp, soft)
    right = _compile(cmp.comparators[0], ctx, xp, soft)
    if isinstance(op, (ast.Gt, ast.GtE)):
        diff = lambda x: left(x) - right(x)
    elif isinstance(op, (ast.Lt, ast.LtE)):
        diff = lambda x: right(x) - left(x)
    else:
        raise ValueError("ind() comparison must be one of < <= > >=")
    k = ctx["sharpness"] / _compare_scale(cmp, ctx["span"])
    if soft:
        return lambda x: jax.nn.sigmoid(k * diff(x))
    return lambda x: (k * diff(x) > 0.0).astype(np.float32)


def _compile(node, ctx: dict, xp, soft: bool) -> Callable:
    """Recursively compile a whitelisted AST node to ``f(x) -> (N,)``."""
    if isinstance(node, ast.Expression):
        return _compile(node.body, ctx, xp, soft)

    if isinstance(node, ast.BinOp):
        L = _compile(node.left, ctx, xp, soft)
        R = _compile(node.right, ctx, xp, soft)
        op = type(node.op)
        if op is ast.Add:
            return lambda x: L(x) + R(x)
        if op is ast.Sub:
            return lambda x: L(x) - R(x)
        if op is ast.Mult:
            return lambda x: L(x) * R(x)
        if op is ast.Div:
            return lambda x: L(x) / R(x)
        if op is ast.Pow:
            return lambda x: L(x) ** R(x)
        raise ValueError(f"operator {op.__name__} not allowed in an equation")

    if isinstance(node, ast.UnaryOp):
        V = _compile(node.operand, ctx, xp, soft)
        if isinstance(node.op, ast.USub):
            return lambda x: -V(x)
        if isinstance(node.op, ast.UAdd):
            return V
        raise ValueError("only unary +/- allowed")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"non-numeric constant {node.value!r}")
        c = float(node.value)
        return lambda x: c

    if isinstance(node, ast.Name):
        if node.id not in ctx["idx_of"]:
            raise ValueError(f"unknown variable {node.id!r}")
        i = ctx["idx_of"][node.id]
        return lambda x: x[:, i]

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id
        if node.keywords:
            raise ValueError(f"{fn}() takes no keyword arguments")
        if fn == "ind" and len(node.args) == 1 \
                and isinstance(node.args[0], ast.Compare):
            return _compile_compare(node.args[0], ctx, xp, soft)
        if fn == "abs" and len(node.args) == 1:
            V = _compile(node.args[0], ctx, xp, soft)
            return lambda x: abs(V(x))
        if fn in ("min", "max") and len(node.args) == 2:
            A = _compile(node.args[0], ctx, xp, soft)
            B = _compile(node.args[1], ctx, xp, soft)
            red = xp.minimum if fn == "min" else xp.maximum
            return lambda x: red(A(x), B(x))
        raise ValueError(f"unsupported call {fn!r} in an equation")

    raise ValueError(
        f"unsupported expression element: {type(node).__name__}")


def compile_expression(expr: str, idx_of: dict, span: dict,
                       sharpness: float):
    """``(soft, hard)`` closures for a bare grammar expression (no residual).

    Same contract as :func:`compile_residual` but for a QUANTITY rather than
    an equation: lets the builder accept ``E[x * y]`` / ``P(x - y > 0)``
    style estimates whose subject is any expression over the variables.
    """
    tree = ast.parse(f"({expr})", mode="eval")
    ctx = {"idx_of": idx_of, "span": span, "sharpness": float(sharpness)}
    soft = _compile(tree, ctx, jnp, True)
    hard = _compile(tree, ctx, np, False)
    return soft, hard


def compile_residual(lhs: str, rhs: str, idx_of: dict, span: dict,
                     sharpness: float):
    """``(soft, hard)`` closures for the residual ``lhs - (rhs)``.

    ``soft`` uses jax + sigmoid indicators (for the differentiable loss);
    ``hard`` uses numpy + exact step indicators (for readouts).  Both accept a
    sample matrix ``x`` of shape ``(N, n_vars)`` and return shape ``(N,)``.
    Raises ``ValueError`` on unknown variables or disallowed syntax — the
    builder catches it and records the estimate in ``skipped``.
    """
    tree = ast.parse(f"({lhs}) - ({rhs})", mode="eval")
    ctx = {"idx_of": idx_of, "span": span, "sharpness": float(sharpness)}
    soft = _compile(tree, ctx, jnp, True)
    hard = _compile(tree, ctx, np, False)
    return soft, hard
