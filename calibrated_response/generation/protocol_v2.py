"""Methodology-v2 elicitation: structure -> tier-1 flood -> case matrix -> battery.

Distilled from the Aug-2026 calibration tests (solver_additions.md §22-23 and
.claude/skills/solver-forecast/SKILL.md). Same enricher shape as protocol.py —
each node maps state -> (new_vars, new_ests, new_reqs), the runner applies an
explicit node list with bounded retries, no solver in the loop — plus
structure fields on the state (cases, outcomes, edges, hub).

What changed vs. the v1 protocols:

- **Full DSL.** Prompts teach everything the parser now accepts: one-sided
  outer relations (hinge penalties), structural equations/inequalities
  including linear floors ``y > a + b*x ~ N(0, s)``, per-estimate ``~ w``
  widths and ``@ n`` pseudo-counts.
- **Two-tier elicitation.** A dedicated flood pass of cheap one-sided truths
  (accounting bounds, prerequisite ceilings, correlation signs, linear
  floors, deliberate tail beliefs) before any point estimates — individually
  true inequalities are mutually consistent for free, and hinges cost
  nothing while satisfied.
- **Case coverage matrix by construction.** Pure code enumerates the cells
  (shock cases x outcome terms, both legs, modal baseline, hub regime rows)
  and the fill pass must answer every cell or mark it ``UNCHANGED`` — an
  empty cell is a decision, never a default. Both resolved calibration
  misses were one silently empty cell.
- **No repeat collapse.** Passes have disjoint scopes and matrix requests
  are key-filtered against existing estimates, so duplicate quantities are
  not generated. (If replicates come back, they get runner-stamped ``@ n``
  budgets — total evidence per quantity capped regardless of draw count —
  instead of a collapse pass.)
- **RAW record.** The directly-stated pre-solver numbers (direct target,
  first marginal per variable) are frozen into the cache payload so fused
  vs. raw can be scored at resolution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from calibrated_response.models.variable import (BinaryVariable,
                                                 ContinuousVariable,
                                                 VariableList)
from calibrated_response.models.query import EqualityProposition
from calibrated_response.models.natural_response import (NaturalEstimateList,
                                                         parse_natural_syntax)
from calibrated_response.generation.protocol import (State, TARGET_NAME,
                                                     key_of_request,
                                                     quantity_key)

CATCHALL_NAME = "unlisted_shock"


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

@dataclass
class StateV2(State):
    """State plus the structure the case matrix is built from."""
    cases: list = field(default_factory=list)      # shock binary names (incl. catch-all)
    outcomes: list = field(default_factory=list)   # resolution-identity terms; [0] = target
    edges: list = field(default_factory=list)      # sorted (a, b) pairs believed dependent
    hub: Optional[str] = None                      # latent intensity hub, if declared
    unchanged: list = field(default_factory=list)  # matrix cells deliberately left at default
    validation: dict = field(default_factory=dict)

    def render_edges(self) -> str:
        if not self.edges:
            return "(none declared)"
        return "\n".join(f"- {a} -- {b}" for a, b in self.edges)

    def render_cases(self) -> str:
        descs = {v.name: v.description for v in self.variables}
        return "\n".join(f"- {c}: {descs.get(c, '')}" for c in self.cases)


# ---------------------------------------------------------------------------
# the grammar the LLM writes in — the full current DSL, plus the calibration
# rules priced in from resolved tests
# ---------------------------------------------------------------------------

GRAMMAR_V2 = """You are an expert forecaster stating calibrated beliefs in a strict grammar.
Each belief is one expression string. Available forms:
- P(x > 5) = 0.3                  probability of an event ('= True'/'= False' for binary variables)
- P(x > 5 | y = True) = 0.4       conditional; multiple conditions comma-separated
- E[x] = 100  /  E[x | y < 2] = 80    expectations, conditional or not
- Corr(x, y) = 0.4                Pearson correlation; NEVER between two binary variables
- The OUTER relation may be one-sided when you only know a ceiling or floor:
  P(x > 5) < 0.1    E[cost] > 100    Corr(x, y) > 0.1
- Structural relations between variables:
  total = a + 0.5*b                exact identity
  total = a + b ~ N(0, 5)          identity with additive noise
  revenue > costs ~ N(0, 5)        inequality (the noise sets boundary softness)
  y > 10 + 2*x ~ N(0, 3)           LINEAR FLOOR: y at least 10 + 2x, pointwise
Optional trailing strength terms, in this order, both optional:
- '~ w'  your uncertainty about the stated number ITSELF (log-odds for P: ~1.5
  weak hunch, ~1.0 moderate, ~0.5 confident; value units for E)
- '@ n'  effective sample size when the number summarizes n observations

Calibration rules (priced in from resolved forecasts — follow them):
- One-sided bounds are free: state them whenever true; they only bind when violated.
- Linear floors: the solver's entropy presses AGAINST the floor, so the fitted
  slope IS your floor. State the honest belief-floor, never a "conservative" low one.
- A year-to-date or cumulative count known near your knowledge cutoff is a LOWER
  BOUND with momentum, not a total — extrapolate the remainder of the period.
- An event "in discussions" / "under consideration" at your cutoff resolves yes
  far above base rate; weight it up, not at the historical prior.
- Variable names must exactly match the provided list.
- Include a brief "logic" explanation for each estimate."""


# ---------------------------------------------------------------------------
# pass A — structure: variables + cases + outcomes + edges (+ optional hub)
# ---------------------------------------------------------------------------

class StructureResponse(VariableList):
    """Pass-A payload: variables (salvage-validated by VariableList) plus the
    structural declarations the matrix is built from."""
    shock_cases: List[str] = Field(
        default_factory=list,
        description="Names of the binary variables that are discrete "
                    "shock/branch hazards (each becomes a case row)")
    outcome_variables: List[str] = Field(
        default_factory=list,
        description="Names of the variables that are terms of the resolution "
                    "identity (what the question is scored on), besides "
                    "'target' itself")
    hub: Optional[str] = Field(
        default=None,
        description="Name of the latent intensity hub variable, if one was "
                    "introduced; null otherwise")
    edges: List[List[str]] = Field(
        default_factory=list,
        description="Pairs [a, b] of variable names you believe are "
                    "dependent (directly influence each other or share a "
                    "cause); pairs not listed are treated as independent")


_STRUCTURE_RULES = """
Rules for the variables:
- Model the MECHANISM, not the process trail: ask "what would have to be true
  of the world for the outcome to happen?" Drivers are quantities with their
  own distributions, not restatements of the outcome at earlier timestamps.
- Every hand-waved assumption becomes a variable with a distribution.
- Add a BINARY SHOCK variable for each discrete hazard that would move the
  drivers (conflict, regulatory action, major outbreak, merger, ...). List
  their names in 'shock_cases'.
- Keep outcome nodes separate from drivers, and list the variables the
  question is scored on in 'outcome_variables'.
- If three or more quantities share one common cause, introduce ONE latent
  unitless hub variable (continuous, lower_bound=-1, upper_bound=4) and name
  it in 'hub'. Its description MUST give an integer intensity legend with
  concrete anchor values, e.g. "0 = recent-normal year (~X), 1 = recent-high
  year (~Y), 2 = severe regime (~Z), 3 = extreme/record regime (~W)".
- Bounds are infinite-confidence estimates: take the most extreme headline
  imaginable and double it. A variable whose most likely value sits at a
  bound is degenerate — widen that side or reformulate (log-scale, binary).
- Write resolution criteria into descriptions: dates, thresholds, data
  source, and what counts, so the variable resolves unambiguously in every
  branch.
- For each variable provide: a short name (2-4 words, underscores), a
  description, binary or continuous, and for continuous a
  lower_bound/upper_bound plus unit.
- Do NOT restate any existing variable (including 'target', the question
  outcome itself).
Also declare 'edges': every pair of variables you believe are dependent."""


async def gen_structure(state: StateV2, client, n: int = 6):
    prompt = (
        f"Question to forecast: {state.question}\n\n"
        f"EXISTING VARIABLES:\n{state.render_variables()}\n\n"
        f"Decompose the question into a mechanism model: up to {n} NEW "
        f"variables, plus the structural declarations described below.\n"
        f"{_STRUCTURE_RULES}\n\n"
        f"Respond in JSON with keys: 'variables', 'shock_cases', "
        f"'outcome_variables', 'hub', 'edges'."
    )
    result = await client.aquery_structured(
        prompt=prompt, response_model=StructureResponse,
        system_prompt="You are an expert forecaster decomposing a prediction "
                      "question into a mechanism model of measurable "
                      "variables, shock cases, and dependencies.",
        temperature=0.7, max_tokens=2000 * n + 8000)
    new = [v for v in result.variables if v.name != TARGET_NAME]
    if not new:
        raise ValueError("no variables survived validation")
    # the catch-all hazard is injected, not elicited: the enumeration itself
    # is the residual risk, and the catch-all bounds it
    new.append(BinaryVariable(
        name=CATCHALL_NAME,
        description="Some other major disruptive event NOT named above "
                    "(a shock outside the enumerated cases) materially "
                    "moves the outcome before resolution."))
    names = {v.name for v in state.variables} | {v.name for v in new}
    binaries = {v.name for v in new if isinstance(v, BinaryVariable)}
    continuous = {v.name for v in new if isinstance(v, ContinuousVariable)}
    state.cases = [s for s in result.shock_cases if s in binaries] \
        + [CATCHALL_NAME]
    state.hub = result.hub if result.hub in continuous else None
    state.outcomes = [TARGET_NAME] + [
        o for o in result.outcome_variables
        if o in names and o != TARGET_NAME]
    edges = set()
    for pair in result.edges:
        if len(pair) < 2:
            continue
        a, b = pair[0], pair[1]
        if a in names and b in names and a != b:
            edges.add(tuple(sorted((a, b))))
    state.edges = sorted(edges)
    return new, [], []


# ---------------------------------------------------------------------------
# pass B — tier-1 flood: cheap one-sided truths
# ---------------------------------------------------------------------------

_TIER1_TASK = """Generate ~{n} TIER-1 statements: beliefs so weak they are
almost certainly TRUE, each cheaply cutting impossible worlds. Only include a
statement if you would bet heavily on it. Draw from every category that
applies:
1. Accounting/ordering inequalities that hold by definition or near-certainly:
   'total_ytd > largest_component ~ N(0, small)', 'revenue > costs ~ N(0, s)'.
2. Prerequisite ceilings — for each necessary precondition d of the outcome:
   'P(target = True | d = False) < 0.02' (one-sided: "this door is closed").
3. Correlation SIGNS for every dependent pair listed under EDGES:
   'Corr(a, b) > 0.1 ~ 0.3' (sign only, wide width). If both variables in a
   pair are binary, use a conditional probability instead.
4. LINEAR FLOORS along edges into continuous outcome variables{hub_clause}:
   'y > a + b*x ~ N(0, s)' — the honest minimum dose-response, in real units.
   Remember: the fitted slope will sit ON your floor, so state the true
   belief-floor, not a low-ball.
5. Deliberate tail beliefs near each continuous variable's bounds, so the
   tail is a stated belief rather than a silent truncation:
   'P(x > v_near_upper_bound) = 0.03 ~ 1.0' (pick the numeric threshold
   yourself, roughly 80-95% of the way to the bound).
Do NOT state unconditional point marginals here (no 'E[x] = v' or
'P(x = True) = p', and no direct 'P(target = True) = p') — a later pass
elicits every marginal; here one-sided bounds and structural relations
only."""


async def gen_tier1(state: StateV2, client, n: Optional[int] = None):
    n = n if n is not None else max(10, 2 * (len(state.variables) - 1))
    hub_clause = ""
    if state.hub:
        hub_clause = (f" (especially dose-response floors coupling every "
                      f"driven quantity to the hub '{state.hub}')")
    prompt = (
        f"Question to forecast: {state.question}\n\n"
        f"AVAILABLE VARIABLES:\n{state.render_variables()}\n\n"
        f"SHOCK CASES:\n{state.render_cases()}\n\n"
        f"EDGES (pairs believed dependent):\n{state.render_edges()}\n\n"
        f"{_TIER1_TASK.format(n=n, hub_clause=hub_clause)}\n\n"
        'Respond with JSON: {"estimates": [{"logic": "...", '
        '"expression": "..."}, ...]}'
    )
    result = await client.aquery_structured(
        prompt=prompt, response_model=NaturalEstimateList,
        system_prompt=GRAMMAR_V2,
        temperature=0.7, max_tokens=500 * n + 6000)
    ests = result.convert_all()
    if not ests:
        raise ValueError("no tier-1 estimates survived parsing")
    return [], ests, []


# ---------------------------------------------------------------------------
# pass C — case coverage matrix: pure-code cells, one fill call
# ---------------------------------------------------------------------------

def propose_matrix_requests(state: StateV2):
    """Pure code, no LLM: enumerate the coverage-matrix cells.

    Rows: each shock case (both legs), the modal baseline (all shocks False,
    when there are >= 2 shocks so it differs from the single False legs),
    and — when a hub exists — its anchor package plus high/low regime rows.
    Columns: the outcome terms. Filtered against existing estimate keys, so
    a cell that tier-1 already covered is not re-asked."""
    binaries = state.binaries()

    def cell(outcome: str, cond: str) -> str:
        if outcome in binaries:
            return f"P({outcome} = True | {cond})"
        return f"E[{outcome} | {cond}]"

    reqs: list[str] = []
    for s in state.cases:
        for o in state.outcomes:
            if o == s:
                continue
            reqs.append(cell(o, f"{s} = True"))
            reqs.append(cell(o, f"{s} = False"))
    if len(state.cases) >= 2:
        baseline = ", ".join(f"{s} = False" for s in state.cases)
        for o in state.outcomes:
            if o not in state.cases:
                reqs.append(cell(o, baseline))
    if state.hub:
        h = next(v for v in state.variables if v.name == state.hub)
        lo, hi = h.lower_bound, h.upper_bound
        span = hi - lo

        def at(frac: float) -> float:
            return round(lo + frac * span, 2)

        # anchor package: mean + graded tail beliefs (under-anchored hubs
        # drift to the box midpoint and drag every coupling with them)
        reqs += [f"E[{state.hub}]",
                 f"P({state.hub} > {at(0.5)})",
                 f"P({state.hub} > {at(0.7)})",
                 f"P({state.hub} < {at(0.2)})"]
        for o in state.outcomes:
            if o == state.hub:
                continue
            reqs.append(cell(o, f"{state.hub} > {at(0.6)}"))
            reqs.append(cell(o, f"{state.hub} < {at(0.4)}"))
    known = {k[:-1] for k in state.estimate_keys()}
    seen: set = set()
    open_reqs: list[str] = []
    for r in reqs:
        k = key_of_request(r)[:-1]
        if k in known or k in seen:
            continue
        seen.add(k)
        open_reqs.append(r)
    return [], [], open_reqs


class MatrixAnswer(BaseModel):
    logic: str = Field(..., description="One-sentence justification")
    answer: str = Field(
        ...,
        description="The quantity copied exactly with '?' replaced by your "
                    "value, optionally with a trailing '~ w' width — OR the "
                    "single word UNCHANGED when you deliberately believe "
                    "this case does not move this quantity")


class MatrixAnswerList(BaseModel):
    answers: List[MatrixAnswer] = Field(default_factory=list)


_MATRIX_TASK = """These are the cells of the CASE COVERAGE MATRIX for this
question: outcome quantities conditioned on each case (shock branches, the
no-shock baseline{hub_note}). Answer EVERY cell, in order:
- Replace '?' with your value. These branch toplines are checksums — the
  gut-level "in a real X, the outcome is over Y" belief. State them even
  when uncertain, with a width: '~ 1.0' log-odds for probabilities, roughly
  half your conditional spread in value units for expectations.
- Budget by SCORE IMPACT, not probability: a 15% branch deserves as much
  care as the 85% one — thinly-estimated shock branches are where forecasts
  die.
- Write UNCHANGED only when you deliberately believe the conditioning case
  does not move the quantity — this is a recorded decision, not a skip."""


async def fill_matrix(state: StateV2, client):
    open_reqs = [
        r for r in state.requests
        if key_of_request(r)[:-1] not in {k[:-1]
                                          for k in state.estimate_keys()}
        and r not in state.unchanged]
    if not open_reqs:
        return [], [], []
    numbered = "\n".join(f"{i + 1}. {r} = ?"
                         for i, r in enumerate(open_reqs))
    hub_note = f", regimes of the hub '{state.hub}'" if state.hub else ""
    prompt = (
        f"Question to forecast: {state.question}\n\n"
        f"AVAILABLE VARIABLES:\n{state.render_variables()}\n\n"
        f"{_MATRIX_TASK.format(hub_note=hub_note)}\n\n"
        f"MATRIX CELLS:\n{numbered}\n\n"
        'Respond with JSON: {"answers": [{"logic": "...", '
        '"answer": "<quantity> = <value> ~ <w>"}, ...]} — one entry per '
        "cell, in order."
    )
    result = await client.aquery_structured(
        prompt=prompt, response_model=MatrixAnswerList,
        system_prompt=GRAMMAR_V2,
        temperature=0.7, max_tokens=400 * len(open_reqs) + 6000)
    if len(result.answers) != len(open_reqs):
        print(f"fill_matrix: {len(result.answers)} answers for "
              f"{len(open_reqs)} cells — aligning by index")
    ests, unchanged, dropped = [], [], []
    for req, ans in zip(open_reqs, result.answers):
        a = ans.answer.strip()
        if a.upper().startswith("UNCHANGED"):
            unchanged.append(req)
            continue
        try:
            ests.append(parse_natural_syntax(a))
        except Exception:
            dropped.append(a[:80])
    if dropped:
        print(f"fill_matrix: dropped {len(dropped)} unparseable "
              f"answer(s): {dropped}")
    if not ests and not unchanged:
        raise ValueError("no matrix cells survived parsing")
    state.unchanged.extend(unchanged)
    return [], ests, []


# ---------------------------------------------------------------------------
# pass D — battery: marginals, spreads, tails, weak direct anchor
# ---------------------------------------------------------------------------

_BATTERY_TASK = """Provide the MARGINAL battery, covering every item below.
Do not restate any existing estimate.
1. REQUIRED, exactly once: the direct estimate 'P(target = True) = p ~ 1.5'.
   The wide '~ 1.5' is deliberate — it is a weak anchor, so the mechanism's
   own answer is allowed to disagree with your gut number.
2. One unconditional marginal for each of these variables (none yet):
{missing}
   Use 'P(name = True) = p ~ 1.0' for binary, 'E[name] = v ~ w' for
   continuous, with w reflecting your real uncertainty.
3. For EACH continuous variable, your 10th/50th/90th percentiles as three
   estimates: 'P(name < q10) = 0.1', 'P(name < q50) = 0.5',
   'P(name < q90) = 0.9' — q10 < q50 < q90 are YOUR values within the
   bounds, reflecting genuine spread (not artificially tight, not centred
   on a round number).
4. For each continuous variable whose upper tail matters, one deliberate
   tail belief near the bound: 'P(name > v) = small ~ 1.0' with v roughly
   80-95% of the way to the upper bound (and the lower tail where relevant),
   so the tail is a stated belief rather than a silent truncation."""


async def gen_battery(state: StateV2, client):
    cont = [v for v in state.variables if isinstance(v, ContinuousVariable)]
    missing = [v.name for v in state.variables
               if v.name != TARGET_NAME
               and not state.has_unconditional(v.name)]
    missing_block = "\n".join(f"   - {m}" for m in missing) or "   (none)"
    n_items = 1 + len(missing) + 4 * len(cont)
    prompt = (
        f"Question to forecast: {state.question}\n\n"
        f"AVAILABLE VARIABLES:\n{state.render_variables()}\n\n"
        f"EXISTING ESTIMATES (context — do not restate):\n"
        f"{state.render_estimates()}\n\n"
        f"{_BATTERY_TASK.format(missing=missing_block)}\n\n"
        'Respond with JSON: {"estimates": [{"logic": "...", '
        '"expression": "..."}, ...]}'
    )
    result = await client.aquery_structured(
        prompt=prompt, response_model=NaturalEstimateList,
        system_prompt=GRAMMAR_V2,
        temperature=0.7, max_tokens=400 * n_items + 6000)
    ests = result.convert_all()
    # no-collapse invariant: never emit a quantity that already has an
    # estimate — k copies act on the solver as one penalty at sd/sqrt(k),
    # a free sharpening (first statement wins; relation ignored so a point
    # belief doesn't slip past an existing point belief as a "new" quantity)
    covered = {k[:-1] for k in state.estimate_keys()}
    kept, dup = [], 0
    for e in ests:
        k = quantity_key(e)[:-1]
        if k in covered:
            dup += 1
            continue
        covered.add(k)
        kept.append(e)
    if dup:
        print(f"gen_battery: dropped {dup} duplicate quantity(ies)")
    if not kept:
        raise ValueError("no battery estimates survived parsing")
    return [], kept, []


# ---------------------------------------------------------------------------
# pass E — pure-code validation + the frozen RAW record
# ---------------------------------------------------------------------------

# identifiers not preceded by a word char or '.' (so '1e3'/'2.5e-1' exponents
# don't read as names), minus grammar keywords/functions
_IDENT = re.compile(r"(?<![\w.])[A-Za-z_]\w*")
_NON_NAMES = {"ind", "abs", "min", "max", "N", "True", "False", "true",
              "false"}


def _idents(expr: str) -> set:
    return set(_IDENT.findall(str(expr))) - _NON_NAMES


def _est_names(est) -> set:
    """Every variable name an estimate references (subjects may be
    arithmetic expressions; equations reference lhs + rhs)."""
    names: set = set()
    if est.estimate_type == "equation":
        return {est.lhs} | _idents(est.rhs)
    if hasattr(est, "proposition"):
        names |= _idents(est.proposition.variable)
    if hasattr(est, "variable"):
        names |= _idents(est.variable)
    if hasattr(est, "variable_a"):
        names |= {est.variable_a, est.variable_b}
    for c in getattr(est, "conditions", []) or []:
        names |= _idents(c.variable)
    return names


def _quantile_thresholds(state: StateV2, var_name: str) -> list:
    """Thresholds t of stated `P(var < t) = p` quantile beliefs."""
    out = []
    for e in state.estimates:
        if e.estimate_type == "probability" and \
                getattr(e.proposition, "proposition_type", "") == "inequality" \
                and e.proposition.variable == var_name and \
                not e.proposition.is_lower_bound and e.relation == "eq":
            out.append((e.probability, e.proposition.threshold))
    return out


def validate(state: StateV2):
    """No LLM: drop estimates referencing unknown names, then check the
    invariants the methodology promises. Hard failures raise (the entry is
    recorded as a failure); soft ones land in state.validation['warnings']
    and ship with the payload."""
    names = {v.name for v in state.variables}
    kept_e, kept_p, dropped = [], [], []
    for est, prov in zip(state.estimates, state.provenance):
        unknown = _est_names(est) - names
        if unknown:
            dropped.append(f"{est.to_query_estimate()} "
                           f"(unknown: {sorted(unknown)})")
        else:
            kept_e.append(est)
            kept_p.append(prov)
    state.estimates, state.provenance = kept_e, kept_p

    errors, warnings = [], []
    covered = {quantity_key(e)[:-1] for e in state.estimates}
    unchanged_keys = set()
    for r in state.unchanged:
        try:
            unchanged_keys.add(key_of_request(r)[:-1])
        except Exception:
            pass

    # direct target (point belief, not just a bound)
    if not any(e.estimate_type == "probability" and e.relation == "eq"
               and isinstance(e.proposition, EqualityProposition)
               and e.proposition.variable == TARGET_NAME
               for e in state.estimates):
        errors.append("missing direct P(target = ...) estimate")

    # matrix coverage on the target row: every case cell is an estimate or
    # a recorded UNCHANGED — never a silent default
    for s in state.cases:
        for cond in (f"{s} = True", f"{s} = False"):
            k = key_of_request(f"P({TARGET_NAME} = True | {cond})")[:-1]
            if k not in covered and k not in unchanged_keys:
                (errors if cond.endswith("True") else warnings).append(
                    f"open matrix cell: P(target | {cond})")

    # hub anchor package (an under-anchored hub drifts to the box midpoint
    # and silently rescales every coupling — measles v0)
    if state.hub:
        if not any(e.estimate_type == "expectation" and e.variable == state.hub
                   for e in state.estimates):
            errors.append(f"hub '{state.hub}' has no E[{state.hub}] anchor")
        n_tails = sum(
            1 for e in state.estimates
            if e.estimate_type == "probability"
            and getattr(e.proposition, "proposition_type", "") == "inequality"
            and e.proposition.variable == state.hub)
        if n_tails < 2:
            warnings.append(f"hub '{state.hub}' has {n_tails} tail "
                            f"belief(s); wants >= 2")

    # per-continuous-variable coverage + bound degeneracy
    for v in state.variables:
        if not isinstance(v, ContinuousVariable):
            continue
        if not state.has_unconditional(v.name):
            warnings.append(f"{v.name}: no unconditional marginal")
        quants = _quantile_thresholds(state, v.name)
        if not quants:
            warnings.append(f"{v.name}: no quantile spread stated")
        span = v.upper_bound - v.lower_bound
        for p, t in quants:
            if abs(p - 0.5) < 0.11 and span > 0:
                frac = (t - v.lower_bound) / span
                if frac < 0.05 or frac > 0.95:
                    warnings.append(
                        f"{v.name}: median {t} sits at {frac:.0%} of the "
                        f"box — bounds look degenerate")

    # coupling: something must link target to the rest of the model
    def _refs_target_plus(e) -> bool:
        ns = _est_names(e)
        return TARGET_NAME in ns and len(ns) > 1
    if not any(_refs_target_plus(e) for e in state.estimates):
        errors.append("no estimate links target to another variable")

    state.validation = {"errors": errors, "warnings": warnings,
                        "dropped": dropped}
    if errors:
        raise ValueError("validation failed: " + "; ".join(errors))
    return [], [], []


def raw_record(state: StateV2) -> dict:
    """The directly-stated pre-solver numbers, frozen for resolution scoring
    (fused vs. RAW is the whole experiment)."""
    raw: dict = {"marginals": {}}
    for e in state.estimates:
        if e.estimate_type == "probability" and e.relation == "eq" and \
                isinstance(e.proposition, EqualityProposition) and \
                e.proposition.variable == TARGET_NAME and \
                "p_target" not in raw:
            p = e.probability
            raw["p_target"] = p if e.proposition.value is True else 1.0 - p
        if e.estimate_type == "probability" and e.relation == "eq" and \
                isinstance(e.proposition, EqualityProposition) and \
                e.proposition.value is True and \
                e.proposition.variable not in raw["marginals"]:
            raw["marginals"][e.proposition.variable] = e.probability
        if e.estimate_type == "expectation" and e.relation == "eq" and \
                e.variable not in raw["marginals"] and \
                not (_idents(e.variable) - {e.variable}):
            raw["marginals"][e.variable] = e.expected_value
    return raw


# name -> (fn, needs_llm), same contract as protocol.ENRICHERS
ENRICHERS_V2 = {
    "gen_structure": (gen_structure, True),
    "gen_tier1": (gen_tier1, True),
    "propose_matrix_requests": (propose_matrix_requests, False),
    "fill_matrix": (fill_matrix, True),
    "gen_battery": (gen_battery, True),
    "validate": (validate, False),
}
