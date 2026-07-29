# Gaussian-KL domain objective

Status: **IMPLEMENTED 2026-07-14** — `domain_prior="gaussian"` /
`prior_bound_sds` in `FlowSamplerModel.constraint_loss` and
`DistributionBuilder`; `--domain-prior gaussian --prior-bound-sds` in
`metaculus/run_flow_solver.py` (recorded in config + manifest); validation
in `tests/maxent_sampler/test_gaussian_kl_prior.py` (6 passed); benchmark
arm `pred_baseline_gk.json` in `metaculus/run_pilot.py`. Uniform remains
the default until benchmarked. The rest of this doc is the design as built.
Origin: the "EXTRA PROGRAMMER NOTE" in
`metaculus/runs/2026-07-14-protocol-pilot/summary.md` — elicitation was
returning degenerate ranges (X in [0, 1] with E[X] = 0), the prompt-side fix
(demand extremely conservative bounds, value interior) landed 2026-07-14, and
this doc is the matching solver-side change.

## Why

The flow solver fits a soft-constrained **maximum-entropy** objective
(`calibrated_response/maxent_sampler/flow_model.py`,
`FlowSamplerModel.constraint_loss`):

    loss(theta) = sum constraint penalties  -  entropy_reg * H(x)

Maximizing entropy on a bounded box is identical (up to an additive
constant) to minimizing KL(p ‖ Uniform(box)): **the implicit default belief
is "uniform over the elicited bounds."** That was tolerable while bounds
were tight — and tight bounds are exactly what produced the degenerate
ranges. Now that the variable-generation prompts demand *extremely
conservative* bounds (true value almost guaranteed interior, mass on both
sides of the central estimate — see `_VAR_RULES` in
`calibrated_response/generation/protocol.py` and the `variable_generation`
prompt in `calibrated_response/generation/prompts.py`), the uniform
reference is actively wrong:

- Any quantity the constraints don't pin gets mean ≈ mid-box and near-uniform
  spread over a deliberately huge range.
- Threshold probabilities P(x > t) default to the *relative measure* of the
  box above t — an artifact of how conservative the bounds happen to be, not
  a belief.
- The entropy bonus rewards pushing mass toward the box edges that the
  elicitor just promised us the value will never reach.

The fix: replace "KL to uniform-over-box" with "KL to a Gaussian centered in
the box." The bounds contract becomes: *bounds are ±k·sd of a default
Gaussian belief*, so making bounds more conservative widens the default
belief proportionally instead of flattening it.

## Objective

Replace the entropy term with a reference-measure KL:

    loss(theta) = sum constraint penalties  +  kl_reg * KL(p_theta ‖ q0)

with a factorized reference over the solver's D sites:

    q0(x) = prod_i q0_i(x_i)
    q0_i  = Normal(mu_i, sd_i)   for continuous variables
    q0_i  = Uniform(0, 1)        for binary sites (see below)

Default reference parameters, derived from the elicited bounds only:

    mu_i = (lo_i + hi_i) / 2
    sd_i = (hi_i - lo_i) / (2 * k),   k = prior_bound_sds, default 2.0

k = 2 means the elicited bounds sit at ±2 sd (~95% central mass), matching
the prompt's "almost guaranteed to fall in the middle portion of the range."
Make k configurable; k → large recovers ~uniform behavior in the box center.

### Exact decomposition (why this is cheap)

    KL(p ‖ q0) = -H(p) - E_p[log q0(x)]

The first term is the **existing exact entropy machinery, unchanged**
(`H(x) = h_const + E[log|det J|]`, see `flow_model.py:65,145`). The second is
a per-sample O(D) reduction on the *same batch* the constraints already
score:

    log q0(x) = sum_cont_i [ -0.5*log(2*pi*sd_i^2) - (x_i - mu_i)^2 / (2*sd_i^2) ]
                + 0                                   # binary sites, log U(0,1) = 0

So the whole change inside `body()` (`flow_model.py:131-148`) is:

```python
# current:
if entropy_reg:
    tot = tot - entropy_reg * (h_const + jnp.mean(ld))
# becomes (ref_mu/ref_sd/ref_mask precomputed jnp arrays, None => old path):
if entropy_reg:
    ent = h_const + jnp.mean(ld)                      # H(p), exact
    if ref_mu is None:
        tot = tot - entropy_reg * ent                 # maxent (KL to uniform)
    else:
        logq0 = jnp.mean(jnp.sum(ref_mask * (
            -0.5 * jnp.log(2 * jnp.pi * ref_sd ** 2)
            - (x - ref_mu) ** 2 / (2 * ref_sd ** 2)), axis=1))
        tot = tot + entropy_reg * (-ent - logq0)      # KL(p ‖ q0)
```

`ref_mask` is 1.0 on continuous sites, 0.0 on binary sites (their Uniform
reference contributes exactly 0 to `log q0`, and their entropy is already
inside `ent` — so binary sites keep their current KL-to-uniform behavior for
free; no special casing beyond the mask).

Notes:

- **Uniform reference is a strict special case.** With q0 = Uniform(box),
  log q0 = -sum log span (a constant), so KL = -H - const: gradients are
  identical to the current objective. Implement as a generalization with
  `reference=None` reproducing today's behavior bit-for-bit; reuse
  `entropy_reg` as the KL weight (natural scale stays 1.0). The loss VALUE
  shifts by a constant between modes — never compare raw losses across modes.
- **Anti-collapse is preserved.** KL(p ‖ q0) = +inf for p with mass on a
  lower-dimensional manifold (the -H term still diverges); the flow's
  "degenerate joints are infinitely penalized" property survives intact.
- **Sign sanity check**: minimizing `-H` spreads mass out; minimizing
  `-H - E_p[log q0]` spreads mass out *while paying quadratically for
  distance from mu* — a soft Gaussian tether, not a hard prior.

## Where everything lives

| concern | file | detail |
|---|---|---|
| loss body to change | `calibrated_response/maxent_sampler/flow_model.py` | `constraint_loss` (~line 100), term at line 144-145; `h_const` built at line 65 |
| domain mapping | same file, `_sample_x_logdet` (line 69) | `x = lower + span * u`, u in (0,1)^D; `self.lower`, `self.span` already jnp arrays — build `ref_mu = lower + 0.5*span`, `ref_sd = span/(2k)` right next to them |
| binary-site identification | `calibrated_response/maxent_sampler/distribution_builder.py` (~line 190) | binaries become `ContinuousVar(name, 0.0, 1.0, 2)`; `self.is_binary` list already exists — pass it (or a mask) down to the model |
| builder plumbing | `distribution_builder.py` `__init__` / `build` (~line 433) | new kwargs `domain_prior="uniform"|"gaussian"` (default `"uniform"` until benchmarked), `prior_bound_sds=2.0`; thread into `FlowSamplerModel.constraint_loss` |
| CLI + provenance | `metaculus/run_flow_solver.py` | `--domain-prior gaussian` + `--prior-bound-sds`; MUST go into the `config` dict so it lands in the per-run `manifest.json` and per-row config stamp |
| readout | `distribution_builder.py` `build_report` (~line 529) | keep `entropy`; optionally add `kl_to_ref` (= `-entropy - E[log q0]`, one extra MC pass) |

## Validation (write these before benchmarking)

1. **Uniform-mode regression**: `domain_prior="uniform"` with a fixed seed
   reproduces today's fits exactly (same predictions on a couple of cached
   entries).
2. **Prior recovery**: no constraints, gaussian mode → fitted marginals of
   continuous vars match N(mu, sd) (mean within ~0.05 sd, p10/p90 within
   ~0.1 sd); binary sites stay ~uniform. (Pattern: `examples/maxent_tests/`
   cases like `case_independent_marginals.py`.)
3. **The degenerate-range scenario that motivated this**: one variable in
   [0, 100], single constraint E[x] = 5. Uniform mode spreads mass across
   the box (P(x > 50) far too high); gaussian mode should keep an
   interior, unimodal marginal near 5. Assert P(x > 50) drops by an order
   of magnitude between modes.
4. **Threshold-default sanity**: variable in [0, 100], no constraint on it;
   P(x > 90) should be ≈ the Gaussian tail (~2% at k=2), not the uniform 10%.

## Benchmark plan

One new solver arm in `metaculus/run_pilot.py` (same cache as the winner,
`--domain-prior gaussian`), scored through `pilot_diagnostics.py --common`
on the standard readout panel (paired ΔBrier vs direct, mean |move|,
max-residual, LTP subset). Free — no re-elicitation. Bump `RUN`.

## Open questions

- **Interaction with the `spreads` elicitation scope** (p10/p50/p90 per
  continuous variable, landed 2026-07-14): spreads enter as *constraints*,
  which should dominate the reference wherever elicited. A tighter variant —
  set (mu, sd) from elicited p50 and (p90-p10)/2.563 instead of from the
  bounds — double-counts those estimates (same information as reference and
  as constraint). Start with bounds-derived reference only.
- **kl_reg schedule**: entropy_reg 1.0 is the natural scale and the pilot
  winner; assume it transfers, but the 0.3 arm from the Phase-1 sweep is
  cheap to re-check in gaussian mode.
- **Truncation**: q0 is an untruncated Gaussian scored on a bounded domain;
  at k = 2 the leaked mass (~5%) only rescales log q0 by a near-constant and
  is not worth correcting. If k is ever pushed below ~1.5, revisit.
