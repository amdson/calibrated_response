# Experiment plan: getting flow > direct

Working doc, 2026-07-12. Supersedes/absorbs `todo.txt`.

## Cost structure (drives the ordering)

- **Solver arms are free**: ~4 s/fit on Colab GPU, reuse the cached pilot
  elicitations. Sweep aggressively.
- **Elicitation variants cost money and fork the cache**: ~$2–4 per
  150-question pilot re-elicitation. Run only when a solver-side result
  demands it, one variant at a time.
- **The full 3,599-question run happens once**, with the recipe that won
  the pilot.

## Fixed readout panel (every experiment reports the same numbers)

From `pilot_diagnostics.py` + `move_analysis`:
1. paired dBrier vs **direct** (primary), vs **market** (reference only —
   the no-retrieval knowledge ceiling is structural)
2. dBrier on the **LTP-violation subset** (n≈18 in pilot) vs consistent
   subset — the joint's only theoretical edge lives here
3. calibration table; move-size distribution (|p_flow − p_direct|)
4. fit health: span-normalised residuals, min p_cond, entropy, credences
   (robust arms), fit seconds

## Levers

### A. Solver-side (free — reuse the existing pilot cache)

| lever | values | status / hypothesis |
|---|---|---|
| `--prob-penalty` | abs / logit (`prob_logit_sd` 0.2/0.3/0.5) | **running** — kills rare-event inflation |
| `--robust` (`p_broken` 0.05/0.15) | on/off | gates should win on LTP subset |
| `entropy_reg` | 1.0 / 0.3 / 0.1 | [todo#3] lower = weaker drift, weaker maxent story |
| `sharpness` | logit: 40/80/160; abs: 20/50 | [todo#3] 80 validated for logit on synthetic |
| anchor asymmetry | direct-target sd × {0.5, 1, 2} vs other estimates | anchor is now **ungated in robust mode** (`anchor_variable`); sd asymmetry still open |
| certainty clip | logit targets clipped to 0.99 / 0.95 | elicited p=1.0 couplings clip at ±logit(0.99)=4.6; tighter clip = weaker overconfident pulls (dzuCO2pcpu case: raw 1.0 targets drove 0.55→0.98) |
| `corr_sd` | 0.1/0.15/0.3 + `--no-corr` | corr currently a no-op; retest after logit |
| fit budget | steps 800/3000, n_samples 2048/8192 | 8192 helps rare-p_cond conditionals |
| fit-seed variance | 5 seeds × 30 questions | [todo#2, cheap half] how much of dBrier is fit noise |

### B. Elicitation-side ($, forks the cache — archive as vN each time)

| lever | values | status / hypothesis |
|---|---|---|
| `--reasoning-effort` | none / low / high | [todo#1] does thinking improve estimate quality & consistency? (LTP-violation rate is the cheap proxy metric) |
| repeat elicitation | k=3 × 50 questions | [todo#2] elicitation-variance vs question-variance → do we need ensembling in the full run? |
| `--n-estimates` | 10 / 15 / 20 | denser coupling; watch LTP rate and p_cond tail |
| `--n-variables` | 4 / 6 | more structure vs more noise |
| model | gemini-flash / gemini-pro | quality ladder; pro ≈ 5–10× cost |
| prompt: probability-scale nudge | on/off | deferred earlier; only worth testing if tails still dominate Brier after logit fix |
| prompt: estimate-type mix | more conditionals vs more corr | after corr retest under logit |

### C. Post-processing (free, applies to any arm)

| lever | notes |
|---|---|
| extremization | one-param logit scaling, fit on half the pilot, score on the other half; improves flow AND direct — run it so the solver comparison is not drowned by a calibration effect |
| flow+direct ensemble | log-odds mean; the boring baseline that often wins — must be in the table |

### D. Measurement (free) [todo#4]

- fit-quality panel on **synthetic ground-truth problems** — already exists
  as `benchmarks/` (constraint reconstruction, held-out NLL); rerun the flow
  engine there after the logit change to confirm no regression
- variance decomposition (A.fit-seed + B.repeat-elicitation) → sample-size
  math for the full run

## Sequence

**Phase 1 — solver sweep on the existing cache (free, this week)**
1. 3-arm penalty check: abs / logit / logit+robust (running).
2. Winner × {entropy_reg 1.0, 0.3} × {robust on/off} — 4 arms, full pilot.
3. Fit-seed variance (5 × 30). Anchor-asymmetry sweep if the joint still
   barely moves.
4. Gate: any arm beating direct at ~2σ overall, or clearly on the LTP
   subset? → that arm is the recipe. If nothing moves the needle, the
   bottleneck is elicitation quality → Phase 2 matters more.

**Phase 2 — elicitation quality ($10–15 total)**
5. Reasoning ladder: none/high on the same 150 pilot ids (2 new caches).
   Compare LTP rate, coverage, dBrier under the Phase-1-winning solver.
6. k=3 repeats × 50 questions → variance decomposition.

**Phase 3 — calibration layer + baselines (free)**
7. Extremization split-half on the best arm; flow+direct ensemble row.

**Phase 4 — the full run**
8. Elicit remaining ~3,450 with winning elicitation config (~$15–40).
9. Solver arms: winner + no-corr ablation (+ robust if not the winner).
10. Headline: paired flow-vs-direct at n≈3,600; LTP-subset effect at
    n≈400; calibration curves.

## Standing cautions

- One change per cache fork; archive caches as `llm_cache_full.vN.json`.
- Equal `steps` across compared arms (the 800-vs-1500 confound from the
  first Colab run).
- Drop stale rows when a predictions file predates a cache regen.
- Provider partials/503s: rerun with `--retry-failures`, never lower the
  bar mid-run.
