"""Composable multi-pass elicitation: enrichers over a shared state.

One uniform node state — (question, variables, estimates, requests) — and a
library of enricher functions that each map state -> new variables and/or
new estimates and/or new requests. A protocol is an explicit list of enricher
applications; the runner merges each delta back with a simple union. This
makes elicitation approaches mix-and-match: fermi breakdowns, correlate
mining, marginal batteries, target couplings, and re-estimation are all the
same shape.

Union semantics (deliberate):
- Estimates are NEVER deduplicated in the state. k estimates of the same
  quantity become k quadratic penalties in the solver — mathematically
  identical to mean-aggregation with sd/sqrt(k), and robust gates can
  convict an outlier repeat individually. The model is the consistency
  device; no aggregation code.
- When estimates are rendered INTO a prompt as data, duplicates collapse to
  one sample per quantity (the first elicited) so the model reads a clean
  belief state.
- Requests are estimates with the value left blank ("P(target = True | x >
  5)"), stored as rendered quantity strings and deduplicated by structural
  key. ``fill_requests`` answers every open request each time it runs, so k
  fill nodes = k independent repeats.

Every enricher is bounded: one LLM call per application (retries live in the
runner), and ``propose_requests`` is pure code. Worst-case calls per question
is len(llm nodes) * (retries + 1), readable off the protocol before launch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from calibrated_response.models.variable import (BinaryVariable,
                                                 ContinuousVariable,
                                                 Variable, VariableList)
from calibrated_response.models.query import (ConditionalProbabilityEstimate,
                                              CorrelationEstimate,
                                              EqualityProposition,
                                              EstimateUnion,
                                              ExpectationEstimate,
                                              InequalityProposition,
                                              ProbabilityEstimate)
from calibrated_response.models.natural_response import (NaturalEstimateList,
                                                         parse_natural_syntax)
from calibrated_response.generation.prompts import format_variables_for_prompt

TARGET_NAME = "target"


# ---------------------------------------------------------------------------
# quantities: the identity of an estimate, independent of its value
# ---------------------------------------------------------------------------

def quantity_str(est: EstimateUnion) -> str:
    """'P(target = True | x > 5.0)' — the estimate minus its value."""
    return est.to_query_estimate().rsplit(" = ", 1)[0]


def quantity_key(est: EstimateUnion) -> tuple:
    """Structural identity: same quantity <=> same key (condition order and
    corr argument order are normalised away)."""
    t = est.estimate_type
    if t == "probability":
        return ("P", est.proposition.to_query_proposition())
    if t == "conditional_probability":
        return ("P", est.proposition.to_query_proposition(),
                tuple(sorted(c.to_query_proposition() for c in est.conditions)))
    if t == "expectation":
        return ("E", est.variable)
    if t == "conditional_expectation":
        return ("E", est.variable,
                tuple(sorted(c.to_query_proposition() for c in est.conditions)))
    if t == "correlation":
        return ("Corr", tuple(sorted((est.variable_a, est.variable_b))))
    raise ValueError(f"unknown estimate type {t!r}")


def key_of_request(expr: str) -> tuple:
    """Key of a rendered quantity string (parse it with a dummy value)."""
    return quantity_key(parse_natural_syntax(f"{expr} = 0.5"))


def negate(prop):
    """The complement event, when it is expressible in the grammar."""
    if isinstance(prop, EqualityProposition) and isinstance(prop.value, bool):
        return EqualityProposition(variable=prop.variable, value=not prop.value)
    if isinstance(prop, InequalityProposition):
        return InequalityProposition(variable=prop.variable,
                                     threshold=prop.threshold,
                                     is_lower_bound=not prop.is_lower_bound)
    return None


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

@dataclass
class State:
    """The one node state every enricher reads and extends."""
    question: str                       # full prompt context for the entry
    variables: list                     # variables[0] is the injected target
    estimates: list = field(default_factory=list)        # EstimateUnion, dupes kept
    provenance: list = field(default_factory=list)       # str per estimate
    requests: list = field(default_factory=list)         # quantity strings
    node_log: list = field(default_factory=list)

    # -- rendering (estimates fed to the model as data are deduped) ---------

    def render_variables(self) -> str:
        info = []
        for v in self.variables:
            d = {"name": v.name, "description": v.description,
                 "type": v.type.value}
            for a in ("lower_bound", "upper_bound", "unit"):
                if hasattr(v, a):
                    d[a] = getattr(v, a)
            info.append(d)
        return format_variables_for_prompt(info)

    def dedup_estimates(self) -> list:
        seen, out = set(), []
        for est in self.estimates:
            k = quantity_key(est)
            if k not in seen:
                seen.add(k)
                out.append(est)
        return out

    def render_estimates(self) -> str:
        ests = self.dedup_estimates()
        if not ests:
            return "(none yet)"
        return "\n".join(f"- {e.to_query_estimate()}" for e in ests)

    # -- membership helpers --------------------------------------------------

    def estimate_keys(self) -> set:
        return {quantity_key(e) for e in self.estimates}

    def has_unconditional(self, var_name: str) -> bool:
        for e in self.estimates:
            if e.estimate_type == "probability" and \
                    e.proposition.variable == var_name:
                return True
            if e.estimate_type == "expectation" and e.variable == var_name:
                return True
        return False

    def binaries(self) -> set:
        return {v.name for v in self.variables
                if isinstance(v, BinaryVariable)}


def merge(state: State, node_tag: str, new_vars, new_ests, new_reqs) -> dict:
    """Union a node's delta into the state. Variables dedupe by name,
    requests by structural key; estimates are appended unconditionally
    (except ill-posed binary-binary correlations, which are dropped)."""
    have = {v.name for v in state.variables}
    n_v = 0
    for v in new_vars or ():
        if v.name not in have:
            state.variables.append(v)
            have.add(v.name)
            n_v += 1
    binaries = state.binaries()
    n_e = 0
    for e in new_ests or ():
        if e.estimate_type == "correlation" and \
                e.variable_a in binaries and e.variable_b in binaries:
            continue
        state.estimates.append(e)
        state.provenance.append(node_tag)
        n_e += 1
    have_req = {key_of_request(r) for r in state.requests}
    n_r = 0
    for r in new_reqs or ():
        k = key_of_request(r)
        if k not in have_req:
            state.requests.append(r)
            have_req.add(k)
            n_r += 1
    return {"vars": n_v, "estimates": n_e, "requests": n_r}


# ---------------------------------------------------------------------------
# enrichers: state -> (new_vars, new_ests, new_reqs)
# ---------------------------------------------------------------------------

_GRAMMAR_SYSTEM = """You are an expert forecaster providing calibrated probability and expectation estimates.
Output estimates using concise mathematical notation with brief reasoning.

ESTIMATE FORMATS:
- Probability: P(variable > threshold) = value  or  P(variable = True) = value
- Expectation: E[variable] = value
- Conditional Probability: P(variable > threshold | condition) = value
- Conditional Expectation: E[variable | condition] = value
- Correlation: Corr(variable_1, variable_2) = value between -1 and 1

RULES:
- Use P(...) for probabilities, E[...] for expectations, Corr(...) for correlations
- Conditions come after | (pipe symbol); multiple conditions separated by commas
- Probabilities must be between 0 and 1
- For binary variables use = True or = False; for continuous use > or < with thresholds
- Corr(...) is ONLY allowed when at least one variable is continuous; relate two
  binary variables with a conditional probability instead
- Variable names must exactly match those provided
- Include a brief "logic" explanation for each estimate"""

_VAR_MODES = {
    "drivers": (
        "Identify {n} variables that could help predict the answer — "
        "measurable or estimable, potentially influential, not perfectly "
        "correlated with each other. Emphasise variables that are necessary "
        "preconditions for the outcome."),
    "fermi": (
        "Break the outcome down like a Fermi estimate: identify {n} variables "
        "that are PREREQUISITES of the outcome (and prerequisites of those "
        "prerequisites) — events or quantities such that the outcome is very "
        "unlikely unless they occur or reach a level. Prefer a chain/tree of "
        "necessary conditions over loosely related context."),
    "correlates": (
        "Identify {n} OBSERVABLE quantities likely to be strongly correlated "
        "with the outcome — indices, counts, prices, event-happened-by-date "
        "facts with checkable historical base rates. Avoid latent abstractions "
        "(momentum, sentiment) that cannot be looked up or estimated directly."),
}

_VAR_RULES = """
For each variable provide: a short name (2-4 words, underscores, no spaces),
a concise description, whether it is binary or continuous, an importance in
[0, 1], and for CONTINUOUS variables a conservative lower_bound/upper_bound
covering all realistic outcomes plus a unit. Scale continuous variables into
roughly the 0.1-10 range when a natural unit allows it.
Do NOT duplicate or restate any existing variable listed above (including
'target', which is the question outcome itself)."""

_EST_SCOPES = {
    "free": (
        "Generate {n} estimates capturing the joint distribution over these "
        "variables: a mix of unconditional and conditional probabilities and "
        "expectations, plus correlations where one variable is continuous. "
        "REQUIRED: include the direct unconditional estimate "
        "P(target = True) = value. REQUIRED: at least 3 estimates must LINK "
        "'target' to other variables."),
    "marginals": (
        "Give exactly ONE unconditional estimate for EACH variable listed "
        "above: P(name = True) = value for binary variables, E[name] = value "
        "for continuous variables. No conditionals, no correlations."),
    "target_connections": (
        "Generate only PAIRWISE estimates linking 'target' to each other "
        "variable: conditional probabilities with target on either side of "
        "the |, or Corr(target, x) for continuous x. For each conditional "
        "P(target = True | c), also give the complementary arm "
        "P(target = True | not-c) (negate the condition: flip True/False, "
        "or flip > to <). Cover every variable at least once."),
    "pairwise": (
        "Generate estimates linking pairs of NON-target variables to each "
        "other — conditional probabilities/expectations or correlations "
        "(with at least one continuous variable). Only include pairs you "
        "genuinely believe are dependent; independent pairs need no "
        "estimate."),
}


async def gen_variables(state: State, client, mode: str = "drivers",
                        n: int = 4):
    prompt = (
        f"Question to forecast: {state.question}\n\n"
        f"EXISTING VARIABLES:\n{state.render_variables()}\n\n"
        f"{_VAR_MODES[mode].format(n=n)}\n{_VAR_RULES}\n\n"
        f"Respond in JSON with a 'variables' list."
    )
    result = await client.aquery_structured(
        prompt=prompt, response_model=VariableList,
        system_prompt="You are an expert forecaster decomposing a prediction "
                      "question into measurable variables.",
        temperature=0.7, max_tokens=2000 * n + 8000)
    new = [v for v in result.variables if v.name != TARGET_NAME]
    if not new:
        raise ValueError("no variables survived validation")
    return new, [], []


async def gen_estimates(state: State, client, scope: str = "free",
                        n: Optional[int] = None):
    n = n if n is not None else max(6, 2 * (len(state.variables) - 1))
    prompt = (
        f"Question to forecast: {state.question}\n\n"
        f"AVAILABLE VARIABLES:\n{state.render_variables()}\n\n"
        f"EXISTING ESTIMATES (do not merely restate these):\n"
        f"{state.render_estimates()}\n\n"
        f"{_EST_SCOPES[scope].format(n=n)}\n\n"
        'Respond with JSON: {"estimates": [{"logic": "...", '
        '"expression": "P(x > 5.0) = 0.3"}, ...]}'
    )
    result = await client.aquery_structured(
        prompt=prompt, response_model=NaturalEstimateList,
        system_prompt=_GRAMMAR_SYSTEM,
        temperature=0.7, max_tokens=500 * n + 6000)
    ests = result.convert_all()
    if not ests:
        raise ValueError("no estimates survived parsing")
    return [], ests, []


def propose_requests(state: State, rules=("direct", "marginals",
                                          "complements", "repeat_target")):
    """Pure code, no LLM: derive open quantities from the current state."""
    reqs: list[str] = []
    if "direct" in rules:
        reqs.append(f"P({TARGET_NAME} = True)")
    if "marginals" in rules:
        for v in state.variables:
            if not state.has_unconditional(v.name):
                reqs.append(f"P({v.name} = True)"
                            if isinstance(v, BinaryVariable) else
                            f"E[{v.name}]")
    if "complements" in rules:
        bounds = {v.name: (v.lower_bound, v.upper_bound)
                  for v in state.variables
                  if isinstance(v, ContinuousVariable)}
        for e in state.estimates:
            if e.estimate_type == "conditional_probability" and \
                    e.proposition.variable == TARGET_NAME and \
                    len(e.conditions) == 1:
                neg = negate(e.conditions[0])
                if neg is None:
                    continue
                if isinstance(neg, InequalityProposition) and \
                        neg.variable in bounds:
                    lo, hi = bounds[neg.variable]
                    # the negated event must have probability mass: skip
                    # complements that fall at/outside the variable's domain
                    # (e.g. negating 'x > 0.0' on a [0, 4] variable)
                    if (neg.is_lower_bound and neg.threshold >= hi) or \
                            (not neg.is_lower_bound and neg.threshold <= lo):
                        continue
                reqs.append(f"P({e.proposition.to_query_proposition()}"
                            f" | {neg.to_query_proposition()})")
    if "repeat_target" in rules:
        for e in state.dedup_estimates():
            names = set()
            if hasattr(e, "proposition"):
                names.add(e.proposition.variable)
            if hasattr(e, "variable"):
                names.add(e.variable)
            if hasattr(e, "variable_a"):
                names |= {e.variable_a, e.variable_b}
            names |= {c.variable for c in getattr(e, "conditions", []) or []}
            if TARGET_NAME in names:
                reqs.append(quantity_str(e))
    return [], [], reqs


async def fill_requests(state: State, client):
    """Answer every open request. Running this node k times = k independent
    repeats of the whole request battery (the union keeps all of them)."""
    if not state.requests:
        return [], [], []
    numbered = "\n".join(f"{i + 1}. {r} = ?"
                         for i, r in enumerate(state.requests))
    prompt = (
        f"Question to forecast: {state.question}\n\n"
        f"AVAILABLE VARIABLES:\n{state.render_variables()}\n\n"
        f"EXISTING ESTIMATES (context; you may revise your view):\n"
        f"{state.render_estimates()}\n\n"
        f"Provide your best estimate for EACH of the following quantities, "
        f"copying the quantity exactly and replacing '?' with your value:\n"
        f"{numbered}\n\n"
        'Respond with JSON: {"estimates": [{"logic": "...", '
        '"expression": "<quantity> = <value>"}, ...]} — one entry per '
        "quantity, in order."
    )
    result = await client.aquery_structured(
        prompt=prompt, response_model=NaturalEstimateList,
        system_prompt=_GRAMMAR_SYSTEM,
        temperature=0.7, max_tokens=400 * len(state.requests) + 6000)
    ests = result.convert_all()
    if not ests:
        raise ValueError("no fills survived parsing")
    return [], ests, []


# name -> (fn, needs_llm). propose_requests is free — it never counts
# against the call budget.
ENRICHERS = {
    "gen_variables": (gen_variables, True),
    "gen_estimates": (gen_estimates, True),
    "propose_requests": (propose_requests, False),
    "fill_requests": (fill_requests, True),
}


def n_llm_calls(protocol: list) -> int:
    """Static worst-case LLM calls per question for one pass (x retries+1)."""
    return sum(1 for name, _ in protocol if ENRICHERS[name][1])
