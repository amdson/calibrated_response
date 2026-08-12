import re
import uuid
from typing import Literal, Union, List
from pydantic import BaseModel, Field, model_validator

from calibrated_response.models.query import (
    Proposition, PropositionUnion, EqualityProposition, InequalityProposition,
    Estimate, EstimateUnion, ProbabilityEstimate, ExpectationEstimate,
    ConditionalExpectationEstimate, ConditionalProbabilityEstimate,
    CorrelationEstimate, EquationEstimate)

# One grammar, shared by the pydantic field pattern (what the LLM is told and
# validated against) and parse_natural_syntax (how it is parsed) so they can
# never drift apart. Deliberately whitespace-tolerant: `P(x>5|y=True)=0.3`
# and `P(x > 5 | y = True) = 0.3` both match.
#
# The relational operator (group 4 below) is `=` for a point belief or
# `< / > / <= / >=` for a one-sided bound (`P(x>5) < 0.3`); the inner `>`/`<`
# of a proposition stays inside the bracketed term, so the two never collide.
#
# A belief may carry two orthogonal trailing strength terms, in this order,
# each optional:
#   `~ w` — noise width: the informant's uncertainty about the estimate
#           ITSELF (`E[Cost] = 100 ~ 20`, `P(A > 10) = 0.3 ~ 1.0`): value
#           units for expectations, log-odds for probabilities, correlation
#           units for Corr.  Fills Estimate.sd, the channel repeat-collapse
#           writes.  (Equations keep their own `~ N(0, sigma)` noise tail —
#           different semantics: spread of the residual, not confidence in
#           a stated number.)
#   `@ n` — effective sample size: how many observations the number
#           summarizes (`E[Cost] = 100 @ 25` = the mean of 25 measurements;
#           `P(A > 10) = 0.6 @ 5` = 3-of-5 frequency).  Fills Estimate.n,
#           the nll penalties' pseudo-count k.
#
# Breakdown:
#   ^\s*([PE])\s*            -> P or E (group 1)
#   [\[\(](.+?)              -> opening ( or [, then the main term (group 2)
#   (?:\s*\|\s*(.+?))?       -> optional | conditions, any spacing (group 3)
#   [\]\)]\s*(<=|>=|<|>|=)\s* -> closing ) or ], then the relation (group 4)
#   (.+?)                    -> the value (group 5)
#   (?:\s*~\s*(...))?        -> optional ~ width (group 6)
#   (?:\s*@\s*(...))?\s*$    -> optional @ sample size (group 7)
_REL_OP = r"(<=|>=|<|>|=)"
_NUM = r"[0-9.eE+\-]+"
_SD_TAIL = (r"(?:\s*~\s*(" + _NUM + r"))?(?:\s*@\s*(" + _NUM + r"))?\s*$")
PE_PATTERN = (
    r"^\s*([PE])\s*[\[\(](.+?)(?:\s*\|\s*(.+?))?[\]\)]\s*" + _REL_OP
    + r"\s*(.+?)" + _SD_TAIL
)
# Corr(X, Y) = r  — scale-free pairwise dependence (also one-sided-bound aware)
CORR_PATTERN = (
    r"^\s*Corr\s*\(\s*([^,|]+?)\s*,\s*([^,|]+?)\s*\)\s*" + _REL_OP
    + r"\s*(.+?)" + _SD_TAIL
)
# Structural relation `lhs <op> rhs [~ N(0, sigma)]`, op one of = < <= > >= — a
# bare-identifier LHS is what separates it from the P()/E[]/Corr() forms (those
# have a bracket right after the head symbol, so they never match a plain
# `name <op>`). An inequality is a one-sided constraint on the same residual:
# `y > 1.5*x` keeps the joint above the line, and the optional noise tail sets
# how soft that boundary is. The rhs is validated loosely here; the real
# arithmetic grammar is enforced at build time by maxent_sampler.equations.
NOISE_PATTERN = r"~\s*N\s*\(\s*0\s*,\s*([0-9.eE+\-]+)\s*\)\s*$"
EQUATION_PATTERN = r"^\s*[A-Za-z_]\w*\s*(?:<=|>=|<|>|=)\s*.+$"
# What the pydantic field admits: any of the forms.
EXPRESSION_PATTERN = (
    f"(?:{PE_PATTERN})|(?:{CORR_PATTERN})|(?:{EQUATION_PATTERN})"
)

# Relational operator -> Estimate.relation (strict/non-strict collapse to the
# same soft bound; the solver's hinge has no notion of a measure-zero boundary).
_OP_TO_RELATION = {"=": "eq", "<": "le", "<=": "le", ">": "ge", ">=": "ge"}


class NaturalEstimate(BaseModel):
    """
    The interface for the LLM.
    It only asks for a strictly formatted string, minimizing token overhead.
    """
    logic: str = Field(..., description="Concise natural language explanation of the estimate.")
    expression: str = Field(
        ...,
        pattern=EXPRESSION_PATTERN,
        description="Format: 'P(A > 10 | B = True) = 0.5' or "
                    "'E[Cost | Tax = True] = 100.0'. The outer relation may also "
                    "be a one-sided bound when you are only confident about a "
                    "ceiling or floor: 'P(A > 10) < 0.2', 'E[Cost] > 100', "
                    "'Corr(A, B) > 0.3'. Use '<'/'>' for bounds and '=' for a "
                    "point estimate. You can also state a structural relation "
                    "linking variables — 'total = 0.4*x + 0.6*y' for an exact "
                    "identity, 'total = 0.4*x + 0.6*y ~ N(0, 5)' with additive "
                    "noise, or a threshold 'hit = ind(total > 100)' — which is "
                    "far better than guessing a correlation when one variable is "
                    "a known function of others. The same form with '<' or '>' "
                    "states an inequality between variables — 'revenue > costs', "
                    "'peak_2030 < capacity ~ N(0, 5)' (the noise sets how soft "
                    "the boundary is) — for bounds one quantity must respect "
                    "relative to another. An estimate may carry trailing "
                    "strength terms, each optional, in this order: '~ w' "
                    "states your uncertainty about the number itself "
                    "('E[Cost] = 100 ~ 20' in value units; "
                    "'P(A > 10) = 0.3 ~ 1.0' in log-odds: ~1.5 weak hunch, "
                    "~0.5 confident), and '@ n' states the effective sample "
                    "size it summarizes ('E[Cost] = 100 @ 25' = mean of 25 "
                    "measurements, 'P(A > 10) = 0.6 @ 5' = a 3-of-5 "
                    "frequency, 'total = a + b ~ N(0, 5) @ 30' = identity "
                    "held on 30 cases). Omit both when unsure — repeated "
                    "estimates of the same quantity also sharpen it "
                    "automatically.",
        examples=["E[battery_cost | growth > 40.0] = 125.0",
                  "P(remaining_slams > 2) < 0.15",
                  "net_worth = tesla_price*musk_shares + spacex_stake"]
    )

    def convert(self) -> EstimateUnion:
        """Helper method to convert this natural response into your structured format."""
        return parse_natural_syntax(self.expression)


class NaturalEstimateList(BaseModel):
    """A list of natural language estimates for LLM generation."""
    estimates: List[NaturalEstimate] = Field(
        default_factory=list,
        description="List of estimates in natural language format"
    )

    @model_validator(mode="before")
    @classmethod
    def _salvage_items(cls, data):
        """LLM-boundary leniency: drop items whose expression fails the
        grammar instead of failing the whole list (one `Corr(X,Y)=0.5` must
        not void nine good estimates). Raises only when nothing survives, so
        callers' retry logic still triggers on genuinely bad responses."""
        if not (isinstance(data, dict) and isinstance(data.get("estimates"), list)):
            return data
        kept, dropped = [], []
        for item in data["estimates"]:
            try:
                kept.append(NaturalEstimate.model_validate(item))
            except Exception:
                expr = item.get("expression", item) if isinstance(item, dict) else item
                dropped.append(str(expr)[:80])
        if dropped:
            print(f"NaturalEstimateList: dropped {len(dropped)} invalid "
                  f"expression(s): {dropped}")
        if not kept and dropped:
            raise ValueError(f"no valid estimates ({len(dropped)} dropped)")
        return {**data, "estimates": kept}

    def convert_all(self) -> List[EstimateUnion]:
        """Convert all natural estimates to structured format."""
        l = []
        for est in self.estimates:
            try:
                l.append(est.convert())
            except Exception as e:
                print(f"Error converting estimate '{est.expression}': {e}")
        return l


# ==========================================
# 2. The Converter Logic
# ==========================================

def _parse_proposition(prop_str: str) -> PropositionUnion:
    """Parses a string like 'X > 5' or 'Y = True' into a Proposition object."""
    # Regex to find operator and split
    # Matches: variable name, operator (>=, <=, >, <, =), value
    match = re.match(r"(.+?)\s*(>=|<=|>|<|=)\s*(.+)", prop_str.strip())
    
    if not match:
        raise ValueError(f"Could not parse proposition: {prop_str}")
        
    var_name, operator, val_str = match.groups()
    var_name = var_name.strip()
    val_str = val_str.strip()

    # Helper to parse values
    def parse_value(v: str) -> Union[bool, float, str]:
        if v.lower() == 'true' or v.lower() == '1': return True
        if v.lower() == 'false' or v.lower() == '0': return False
        try: return float(v)
        except ValueError: return v  # Return as string if not float/bool

    value = parse_value(val_str)

    if operator == "=":
        # Construct EqualityProposition
        assert isinstance(value, (bool, str)), "EqualityProposition value must be bool or str"
        return EqualityProposition(
            variable=var_name,
            value=value
        )
    else:
        # Construct InequalityProposition
        assert isinstance(value, (int, float)), "InequalityProposition value must be numeric"
        return InequalityProposition(
            variable=var_name,
            threshold=float(value), # Inequalities require floats in your model
            is_lower_bound=(not operator in ["<", "<="])
        )

def parse_natural_syntax(expression: str) -> EstimateUnion:
    """
    Converts a natural language math string into the complex EstimateUnion structure.
    """
    # 1. Regex Extraction (same grammar the pydantic field validates against)
    corr = re.match(CORR_PATTERN, expression)
    if corr:
        var_a, var_b, op, value_str, sd_str, n_str = corr.groups()
        return CorrelationEstimate(
            id=f"C_{var_a.strip()[:12]}_{var_b.strip()[:12]}",
            variable_a=var_a.strip(),
            variable_b=var_b.strip(),
            correlation=float(value_str.strip().rstrip(".")),
            relation=_OP_TO_RELATION[op],
            sd=float(sd_str) if sd_str else None,
            n=float(n_str) if n_str else None,
        )

    match = re.match(PE_PATTERN, expression)

    if not match:
        # Not a P()/E[]/Corr() form — try a structural equation `lhs = rhs`,
        # peeling off the optional `@ n` count and `~ N(0, sigma)` noise
        # tails first (in that order: `lhs = rhs ~ N(0, 5) @ 30`).
        body = expression.strip()
        eq_n = None
        m_n = re.search(r"@\s*(" + _NUM + r")\s*$", body)
        if m_n:
            eq_n = float(m_n.group(1))
            body = body[: m_n.start()].strip()
        noise_sd = None
        m_noise = re.search(NOISE_PATTERN, body)
        if m_noise:
            noise_sd = float(m_noise.group(1))
            body = body[: m_noise.start()].strip()
        m_eq = re.match(r"^\s*([A-Za-z_]\w*)\s*(<=|>=|<|>|=)\s*(.+?)\s*$", body)
        if m_eq:
            lhs, op, rhs = m_eq.group(1), m_eq.group(2), m_eq.group(3).strip()
            return EquationEstimate(
                id=f"EQ_{lhs[:16]}",
                lhs=lhs,
                rhs=rhs,
                noise_sd=noise_sd,
                relation=_OP_TO_RELATION[op],
                n=eq_n,
            )
        raise ValueError(f"Invalid syntax: {expression}")

    type_char, main_term, condition_str, op, value_str, sd_str, n_str = \
        match.groups()
    relation = _OP_TO_RELATION[op]

    # 2. Parse Value (tolerate a trailing period / percent phrasing slip)
    est_value = float(value_str.strip().rstrip("."))
    est_sd = float(sd_str) if sd_str else None
    est_n = float(n_str) if n_str else None
    
    # 3. Parse Conditions (if any)
    conditions = []
    if condition_str:
        # Split by comma if multiple conditions exist (e.g. "A=1, B>2")
        cond_parts = [c.strip() for c in condition_str.split(',')]
        conditions = [_parse_proposition(c) for c in cond_parts]

    # 4. Generate a unique ID (required by your base class)
    est_id = f"{type_char}_{main_term}_{'_'.join([c.variable[:5] for c in conditions])}"

    # 5. Construct the specific Object
    if type_char == 'P':
        # Probability: Main term is a Proposition (e.g., P(X > 5))
        main_prop = _parse_proposition(main_term)
        
        if conditions:
            return ConditionalProbabilityEstimate(
                id=est_id,
                proposition=main_prop,
                conditions=conditions,
                probability=est_value,
                relation=relation,
                sd=est_sd,
                n=est_n
            )
        else:
            return ProbabilityEstimate(
                id=est_id,
                proposition=main_prop,
                probability=est_value,
                relation=relation,
                sd=est_sd,
                n=est_n
            )

    elif type_char == 'E':
        # Expectation: Main term is just a Variable (e.g., E[Cost])
        variable_name = main_term.strip()

        if conditions:
            return ConditionalExpectationEstimate(
                id=est_id,
                variable=variable_name,
                conditions=conditions,
                expected_value=est_value,
                relation=relation,
                sd=est_sd,
                n=est_n
            )
        else:
            return ExpectationEstimate(
                id=est_id,
                variable=variable_name,
                expected_value=est_value,
                relation=relation,
                sd=est_sd,
                n=est_n
            )
            
    raise ValueError(f"Unknown estimate type: {type_char}")