# Protocol pilot (2026-07-14): the echo bug, and what to change

> **STATUS (2026-07-14): all recommended changes landed** — see the
> 2026-07-14 entry in `experiments.md`. This doc is kept as the summary
> of the (void) run whose outputs sit in this directory. `pred_baseline`
> and `pred_v1` were deleted (resume-contaminated); the echo-bug caches
> moved to `caches/archive/echo-bug-2026-07-14/`.

Five arms fit on Colab/GPU from four elicitation caches, scored on the 29
questions fit in all of them (`pilot_diagnostics.py --common`, base rate 0.31).

| arm | Brier flow | its own direct-LLM | paired ΔBrier (flow − direct) |
|---|---|---|---|
| baseline | 0.1416 | 0.1407 | +0.0009 ± 0.0015 |
| v1 | 0.1764 | 0.1574 | **+0.0190** ± 0.0287 |
| v1x2 | **0.1378** | 0.1495 | −0.0118 ± 0.0093 |
| v1x2_collapsed | 0.1384 | 0.1495 | −0.0111 ± 0.0094 |
| v1_fermi | 0.1535 | 0.1541 | −0.0005 ± 0.0017 |

Everything beats a constant base rate (0.214); everything loses to the market
(0.0790 on the 13 market questions).

**None of these rankings mean anything yet.** The multi-pass protocols carry a
bug that double-counts the direct target estimate, which pins the solver to the
LLM's own answer. Fix that before reading anything into the table.

---

## 1. The echo bug

`fill_requests` is asked to answer quantities that earlier passes **already
estimated**, and `State.render_estimates()` shows it those estimates in its
prompt. So it copies the number it can see. Every duplicate group in every
multi-pass cache is cross-node — `gen_estimates[...] + fill_requests`:

```
llm_cache_v1.json        356  gen_estimates[target_connections] + fill_requests
                          60  gen_estimates[marginals]          + fill_requests
llm_cache_v1x2.json      193  gen_estimates[target_connections] + fill_requests + fill_requests
                          32  gen_estimates[marginals]          + fill_requests + fill_requests
llm_cache_v1_fermi.json  351  gen_estimates[target_connections] + fill_requests
                          57  gen_estimates[marginals]          + fill_requests
```

The copies are *identical*, not re-sampled — median log-odds spread within a
group is **0.000** for v1 and v1_fermi, 0.039 for v1x2. Redundancy is 1.62× in
v1 (416 groups of k=2) and 2.27× in v1x2 (225 groups of k=3).

Worst of all, the **direct target estimate is duplicated in every single entry**:

```
baseline   copies of the direct target estimate -> {1: 60}   duplicated in  0/60
v1                                              -> {2: 60}   duplicated in 60/60
v1x2                                            -> {3: 32}   duplicated in 32/32
v1_fermi                                        -> {2: 57}   duplicated in 57/57
```

### Why it wrecks the solver

`merge()` appends estimates unconditionally (`state.estimates`, "dupes kept")
and `serialize_state()` persists that raw list — `dedup_estimates()` exists but
is only used to *render prompts*. So the solver receives k copies and applies k
independent quadratic penalties. k penalties on one quantity ≡ **one penalty at
√k the weight**, i.e. effective sd/√k. The anchor therefore gets

- baseline: sd 0.3 (one copy)
- v1, v1_fermi: 0.3/√2 ≈ 0.21
- v1x2: 0.3/√3 ≈ 0.17

...while every other constraint keeps its nominal width. The anchor is
systematically over-weighted in exactly the arms that are supposed to *beat* it.

The prediction this makes is confirmed: median `|flow − direct|` is **0.004–0.009
in all five arms**. The joint model is reproducing the LLM's own answer to within
a percentage point on the median question. Whatever the extra passes learn, the
over-weighted anchor drowns it — and in v1 the extra constraints only add tension
(median residual 0.116, p90 0.443, vs baseline's 0.064 / 0.140), which is why v1
lands *worse than its own direct estimate*.

### The deeper point about repeats

Repeats elicited this way can never work. The fill pass sees its own prior answer
in context, so it echoes. A repeat only carries information if it is drawn in a
**fresh context with the prior estimates hidden**. Right now v1x2 is paying for a
second LLM call and buying a copy.

---

## 2. `--collapse-repeats` doesn't rescue it — my floor is wrong

The collapsed arm moved nothing (0.1378 → 0.1384; median |Δp| 0.0035). Structural
reason, in `protocol.collapse_repeats`:

```python
sd = max(prob_logit_sd / math.sqrt(k), statistics.pstdev(ls))   # <-- bug
```

For an identical duplicate the spread is 0, so this returns `0.3/√k` — **exactly
the naive √k sharpening it was written to prevent**. It only widens where the
spread beats that floor: 11% of v1x2 groups, 7% of v1's.

The `/√k` term is indefensible. `prob_logit_sd` encodes *how much we trust this
claim about the world*; sampling the LLM more times does not make its claim more
reliable, and for an echo it isn't even sampling. Correct rule:

```python
sd = max(prob_logit_sd, statistics.pstdev(ls))
```

Duplicates count once. Disagreement widens past single-estimate trust. Nothing
ever sharpens for free.

---

## 3. Recommended changes, in order

1. **Stop re-requesting known quantities.** Filter `propose_requests` output
   against `state.estimate_keys()`. The `direct`, `marginals`, and
   `repeat_target` rules currently re-request quantities that already have
   estimates; only `complements` yields genuinely new keys. This removes the echo
   at the source and is the only change that requires re-elicitation.

2. **Fix the collapse floor** to `max(prob_logit_sd, pstdev)` (above), and make
   collapse **the default in the solver rather than opt-in** — a `--no-collapse`
   escape hatch instead of `--collapse-repeats`. Defense in depth: even a cache
   that still contains duplicates then cannot √k-sharpen the solver. Free, no
   re-elicitation.

3. **Refit `baseline` and `v1` from scratch on GPU.** Both files are
   resume-contaminated: 27/29 of `pred_baseline`'s common rows and 14/29 of
   `pred_v1`'s are stale CPU fits from before the push (60s fit times vs 4s give
   it away) — Colab's `git reset --hard` restored the committed prediction files
   and the resume-skip kept them. The solver config is semantically identical (the
   per-estimate-sd change is a strict no-op when `est.sd is None`, verified
   against the diff), so this is a backend/RNG difference, not a config confound —
   but `baseline` is ~93% CPU-fit *and* is the arm that looks competitive, so its
   margin is not trustworthy. Delete the two files and refit; costs nothing (no
   LLM calls).

4. **If repeats are still wanted, re-elicit them in a fresh context** with prior
   estimates hidden, so the draw is independent. Otherwise drop v1x2 — at present
   it is a second API call that returns a copy.

5. Re-run the 5-arm comparison. Until 1–3 land, the protocol ranking above is
   measuring anchor over-weighting plus one lucky 32-entry cache.

---

## 4. Note: the JSON files need organizing

36 JSON files sit flat in `metaculus/`, spanning five months and four
generations of experiment, with no convention distinguishing inputs from
`$`-expensive caches from free regenerable outputs. This directly caused the
stale-row contamination in item 3: predictions files are committed, so a resume
silently reuses rows fit by older code.

Proposed layout:

```
metaculus/
  data/                     inputs, rarely change, not regenerable
      full_dataset.json  tiny_dataset.json  pilot_ids.txt  pilot_ids_small.txt
  caches/                   elicitation output — costs $, treat as IMMUTABLE
      v1/  llm_cache_v1.json  llm_cache_v1.json.failures.json
      ...
      archive/            superseded cache forks (llm_cache_full.v0/.v1, llm_cache_n20, ...)
  runs/                     solver output — free, regenerable, DISPOSABLE
      2026-07-12-penalty-arms/   arm_abs.json arm_logit.json arm_logit_robust.json
      2026-07-13-density-sweep/  arm_n20_e5.json ... arm_n20_e20_robust.json
      2026-07-14-protocol-pilot/ pred_*.json
      archive/            pre-flow relics (baseline_predictions*, maxent_smm_*, flow_predictions*)
```

Rules worth adopting with it:

- **One run directory per solver run, named by date + slug.** A fresh directory
  means resume can never cross a code change — which is the actual mechanism
  behind item 3, not a one-off mistake.
- **`runs/` is regenerable; consider gitignoring it** (or committing only a
  `summary.md` per run). Caches are the thing that costs money and must stay
  tracked.
- Each run dir gets the solver config + the git SHA it was fit at. `config` is
  already stamped per row; promote it to a `manifest.json` so a run can be
  audited without loading a prediction file.
- Keep `experiments.md` as the results log; this doc folds into it once the
  changes land.

Existing "Standing cautions" in `experiments.md` already say *"drop stale rows
when a predictions file predates a cache regen"* — the layout above is how that
caution stops depending on anyone remembering it.

PROGRAMMER NOTE
I like this advice, I'd also like to implement it as on of the elicitation functions. --- Elicit spreads, not point expectations. E[cavity_reduction] = 48 tells the solver nothing about whether it clears 50 — that's decided entirely by the maxent default width. For every threshold question, the spread is the answer, and it's currently the one thing you never ask for. Ask for p10/p50/p90 per leaf variable. 

EXTRA PROGRAMMER NOTE
It's a problem that the elicitation functions are giving me variables with degenerate ranges. E.g. a X in [0, 1] with E(X)=0. I'm thinking we should ask the language models for extremely conservative bounds, such that the variable is almost guaranteed to be within the middle range of the bounds. We can then optionally apply a gaussian KL penalty on the domain of the solver instead of simple entropy.

> LANDED 2026-07-14: prompt side done — `_VAR_RULES` (protocol.py) and the
> variable_generation prompt (prompts.py) now demand extremely conservative
> bounds with the true value interior and mass on both sides, and tell the
> model to reformulate quantities pinned at a physical limit. The gaussian
> KL domain penalty in the solver is NOT implemented — still open.