"""Regularizers for tensor-chain constraint fitting (spec §6, TN version).

Constraint matching pins only the *specified* marginals; the rest of the
distribution is free, and unregularised it collapses onto a few bins (see the
pairwise plot). These regularizers shape the under-determined part. They act on
the model's exact per-site marginals ``p(X_i)`` (differentiable via
:meth:`TensorChain.site_marginal`) or directly on the cores.

All take ``(model, params) -> scalar`` (lower = better), so they compose:

    from calibrated_response.tn import losses as L
    loss = L.combined_loss(model, constraints,
                           [(L.marginal_curvature, 5.0), (L.uniform_coverage, 0.1)])

``REGULARIZERS`` is a name->fn registry for quick sweeping.
"""

from __future__ import annotations

import jax.numpy as jnp

_EPS = 1e-12


def _site_marginals(model, params):
    return [model.site_marginal(params, i) for i in range(model.n)]


# ---------------------------------------------------------------------------
# regularizers  (each: (model, params) -> scalar, lower is better)
# ---------------------------------------------------------------------------

def l2_cores(model, params):
    """Shrink core magnitudes — a mild anti-degeneracy prior on the amplitudes."""
    cs = params["cores"]
    return sum(jnp.mean(c ** 2) for c in cs) / len(cs)


def neg_marginal_entropy(model, params):
    """``-H(p(X_i))`` averaged over sites: minimise to *maximise* marginal entropy.

    Pushes each per-site marginal toward uniform (spreads mass) without caring
    about bin adjacency.
    """
    tot = 0.0
    for p in _site_marginals(model, params):
        tot = tot + jnp.sum(p * jnp.log(p + _EPS))          # = -H(p)
    return tot / model.n


def neg_renyi2_entropy(model, params):
    """``-H2(p)/n`` — minimise to *maximise* the joint Rényi-2 entropy.

    The joint counterpart of :func:`neg_marginal_entropy` (and the ``-beta*H2``
    entropy-pressure term of ``smothing_notes.md``): ``H2 = -log sum_x p(x)^2``
    is maximised only by the uniform over *full configs*, so it penalises
    phantom correlations between unconstrained variables — structure every
    per-site marginal regularizer is blind to (a chain can have perfectly
    uniform marginals while adjacent sites are strongly coupled). Exact and
    differentiable via :meth:`TensorChain.renyi2_entropy`; the born kind pays a
    four-copy contraction (steep in ``r``), the nonneg kind a two-copy one.
    Normalised per site so weights transfer across chain sizes.
    """
    return -model.renyi2_entropy(params) / model.n


def uniform_coverage(model, params):
    """Cross-entropy ``H(uniform, p) = -mean_k log p_k`` averaged over sites.

    Penalises *any* near-empty bin (a log barrier against zeros), so it fills
    troughs more aggressively than entropy. The TN analog of
    ``pc.losses.uniform_coverage_regularizer``.
    """
    tot = 0.0
    for p in _site_marginals(model, params):
        tot = tot + (-jnp.mean(jnp.log(p + _EPS)))
    return tot / model.n


def marginal_tv(model, params):
    """Total variation ``sum_k |p_{k+1}-p_k|`` averaged over sites.

    Encodes that neighbouring bins of a *continuous* variable should have
    similar mass — a smoothness prior that entropy/coverage (adjacency-blind)
    do not provide. Favours piecewise-flat marginals.
    """
    tot = 0.0
    for p in _site_marginals(model, params):
        tot = tot + jnp.sum(jnp.abs(jnp.diff(p)))
    return tot / model.n


def marginal_curvature(model, params):
    """Squared second difference ``sum_k (p_{k+1}-2p_k+p_{k-1})^2`` per site.

    A stronger smoothness prior than TV: penalises curvature, favouring gently
    varying (rather than merely flat) marginals. Good default for discretised
    continuous variables.
    """
    tot = 0.0
    for p in _site_marginals(model, params):
        d2 = p[2:] - 2.0 * p[1:-1] + p[:-2]
        tot = tot + jnp.sum(d2 ** 2)
    return tot / model.n


def _phys_axis(i, n):
    """Physical (bin) axis of core i: (d,r)->0, (r,d,r)->1, (r,d)->1."""
    return 0 if i == 0 else 1


def core_tv(model, params):
    """L2 first-difference of each core *along the physical (bin) axis*.

    A smoothness penalty applied **directly to the tensor weights** rather than the
    derived marginals — no contraction, just finite differences on the raw cores,
    so it is far cheaper and scales to large ``n``. It smooths the *amplitude*
    across adjacent bins (discouraging sign flips / spikes between neighbouring
    values), which pushes the density smooth indirectly. Bond axes have no natural
    order, so only the physical axis is differenced.

    Caveat: this acts on the parameterisation, not the distribution — MPS gauge
    freedom means the same ``p`` can have rougher/smoother cores, so unlike
    :func:`marginal_curvature` it is not a pure function of the density (it also
    biases toward a smooth-amplitude gauge). Cheap and effective in practice.
    """
    cores = params["cores"]
    tot = 0.0
    for i, c in enumerate(cores):
        tot = tot + jnp.mean(jnp.diff(c, axis=_phys_axis(i, model.n)) ** 2)
    return tot / len(cores)


def core_curvature(model, params):
    """L2 second-difference of each core along the physical axis (see :func:`core_tv`).

    Penalises curvature of the amplitude across bins. In practice **weaker than
    :func:`core_tv`**: the second difference is easy to zero out under the MPS gauge
    without smoothing the density (the roughness hides in signs / bond structure),
    so it barely moves the marginals. Prefer ``core_tv`` for a direct-weight
    smoother; kept here for comparison.
    """
    cores = params["cores"]
    tot = 0.0
    for i, c in enumerate(cores):
        tot = tot + jnp.mean(jnp.diff(c, n=2, axis=_phys_axis(i, model.n)) ** 2)
    return tot / len(cores)


REGULARIZERS = {
    "l2_cores": l2_cores,
    "entropy": neg_marginal_entropy,
    "renyi2": neg_renyi2_entropy,
    "coverage": uniform_coverage,
    "tv": marginal_tv,
    "curvature": marginal_curvature,
    "core_tv": core_tv,
    "core_curvature": core_curvature,
}

# ---------------------------------------------------------------------------
# projection regularizer on Y = a·X  (needs config -> a factory, not in the
# arg-free REGULARIZERS registry)
# ---------------------------------------------------------------------------

def linear_isotropy(n_dirs=8, target=1.0, seed=0, within_bin=False):
    """Projection-isotropy penalty ``E_a[(Var(a·X)/||a||^2 - target)^2]`` (spec §6).

    Draws ``n_dirs`` random directions ``a`` and penalises the spread of the
    projection ``a·X`` two-sidedly, using the exact
    :meth:`TensorChain.linear_moments` sweep (so it scales to large ``n``). Guards
    against collapse / runaway spread of the joint along random directions —
    structure the per-site marginals cannot see. Returns a ``(model, params) ->
    scalar`` closure; compose it like any regularizer:

        loss = L.combined_loss(model, cons, [(L.linear_isotropy(target=800.), 1e-4)])

    ``target`` is the desired ``Var(a·X)/||a||^2`` (problem-scale dependent — e.g.
    for uniform spread over ``[0, 100]`` it is ~ 100^2/12 ≈ 833).
    """
    import numpy as np

    def reg(model, params):
        dirs = np.random.default_rng(seed).normal(size=(n_dirs, model.n)).astype(np.float32)
        tot = 0.0
        for a in dirs:
            _, var = model.linear_moments(params, a, within_bin=within_bin)
            tot = tot + (var / float(np.sum(a ** 2)) - target) ** 2
        return tot / n_dirs

    return reg


def amplitude_roughness(sites=None, order=2):
    """Gauge-invariant amplitude-curvature penalty ``mean_i <psi|L_i|psi>/Z`` (notes).

    The faithful implementation of ``smothing_notes.md`` under global (non-sweeping)
    optimisation: a two-copy contraction inserting the roughness metric
    ``L = D^T D`` at each smoothed site (see :meth:`TensorChain.amplitude_roughness`).
    Because it is a pure function of the amplitude ``psi`` (self-normalised by
    ``<psi|psi>``), it is invariant to MPS gauge and to rescaling — so, unlike the
    cheap :func:`core_curvature`/:func:`core_tv` (which act on the raw cores in an
    arbitrary gauge and are partly gauge-escapable), the optimiser cannot satisfy it
    by moving gauge without actually smoothing the represented function. ``order=2``
    penalises curvature, ``order=1`` slope. ``sites=None`` smooths every site.

    Cost is one doubled contraction per smoothed site (``O(n) `` of them), i.e. the
    same order as :func:`marginal_curvature`, but it targets the *amplitude* curvature
    the notes recommend rather than the density marginal. Returns a
    ``(model, params) -> scalar`` closure.
    """
    def reg(model, params):
        return model.amplitude_roughness(params, sites=sites, order=order)

    return reg


def marginal_kl(sites, ref, direction="forward", weight=1.0):
    """Pull the joint marginal over ``sites`` toward reference ``ref`` in KL.

    A composable wrapper over :meth:`TensorChain.marginal_kl` — use it as either a
    ``combined_loss`` term or (equivalently) a ``("kl", sites, ref)`` entry in the
    constraint list. ``sites`` is an int or tuple; ``ref`` is any array shaped like
    the joint marginal (renormalised internally). ``direction="forward"`` is
    ``KL(ref || p)`` (fit-to-target); ``"reverse"`` is ``KL(p || ref)``. Returns a
    ``(model, params) -> scalar`` closure.
    """
    def reg(model, params):
        return weight * model.marginal_kl(params, sites, ref, direction=direction)

    return reg


def expectation_target(values, target, weight=1.0):
    """Squared-error pull of ``E[g(X)]`` toward ``target`` (see :meth:`TensorChain.expectation`).

    ``values`` is either a ``{site: g}`` dict (separable ``E[sum_i g_i(X_i)]``) or a
    ``(sites, table)`` tuple (joint ``E[g(X_sites)]``). Composable like any
    regularizer, or equivalently an ``("expect", values, target)`` constraint
    entry. Returns a ``(model, params) -> scalar`` closure.
    """
    def reg(model, params):
        return weight * (model.expectation(params, values) - target) ** 2

    return reg


def spike_slab_prior(centers, target, spike_sd, slab_sd, spike_w=0.85):
    """Spike-and-slab prior over a latent-target site's bins (narrow + broad normal).

    A differentiable stand-in for a true spike-and-slab: a ``spike_w`` mixture of a
    *narrow* normal at ``target`` (you believe the constraint) and a *broad* normal
    (the constraint may be off, or bogus). ``centers`` are the latent site's bin
    centres (``model.disc.bin_centers(c_site)``). Proper by construction once
    renormalised over the finite binned domain. Returned as a plain ``float32``
    array so you can inspect / plot it, or feed it straight to a ``("kl", c, prior)``
    constraint if you only want the prior half.
    """
    import numpy as np
    c = np.asarray(centers, np.float64)
    g = lambda sd: np.exp(-0.5 * ((c - target) / sd) ** 2)
    p = spike_w * g(spike_sd) + (1.0 - spike_w) * g(slab_sd)
    return (p / p.sum()).astype(np.float32)


def cond_expectation_target(site, given, target, f=None, weight=1.0):
    """Squared-error pull of ``E[f(X_site) | given]`` toward ``target``.

    Composable wrapper over :meth:`TensorChain.cond_expectation`, equivalent to a
    ``("cond_expect", site, given, target)`` constraint entry. ``given`` is an event
    ``{site: mask}`` (see :meth:`TensorChain.threshold_mask`); ``f`` a callable on
    ``site``'s bin centres (``None`` => conditional mean). Returns a
    ``(model, params) -> scalar`` closure.
    """
    def reg(model, params):
        return weight * (model.cond_expectation(params, site, given, f=f) - target) ** 2

    return reg


def robust_expectation(x_site, c_site, target, f=None, given=None, *, spike_sd=None,
                       slab_sd=None, spike_w=0.85, w_prior=1.0, w_couple=1.0):
    """Robust (spike-and-slab) constraint ``E[f(X)|given] ~ target`` via a latent target site.

    Instead of a hard ``(E[f(X)] - target)^2``, treat the target itself as an
    uncertain latent ``c`` (an extra :func:`~calibrated_response.tn.latent_var` site
    on the chain, adjacent to ``x_site``). Everything is read off joint marginals of
    ``(x, c)`` (:meth:`TensorChain.joint_marginal`):

    - ``p(c)`` is pulled toward a :func:`spike_slab_prior` centred at ``target`` —
      your meta-uncertainty about the constraint (``w_prior``).
    - ``E[f(X)|c[, given]]`` is pulled toward ``c``, written division-free as the
      **mass-weighted** residual ``(f_vals @ J) - centers_c * p_c`` (which equals
      ``p_c * (E[f(X)|c] - c)``), so slab bins / out-of-event mass with ~no weight
      cost nothing and there is no divide-by-``p_c`` (``w_couple``).

    Marginalising ``c`` out at prediction time yields a *mixture*: mostly the
    ``E[f(X)]=target`` solution (spike weight), plus a broad escape component (slab)
    that a contradicting constraint can be explained away into rather than corrupting
    the fit. With ``w_prior`` small the spike/slab weight is itself learned — a
    credence in the constraint. See the discussion in the notebook / chat.

    Args:
      x_site, c_site: chain sites of the data variable and its latent target.
      target:         believed value of ``E[f(X)|given]`` (the spike location).
      f:              callable applied to ``x``'s bin centres (``None`` => identity,
                      i.e. a mean constraint). E.g. ``lambda x: x**2`` for ``E[X^2]``.
      given:          optional event ``{site: mask}`` making it a *conditional*
                      expectation constraint (the coupling is enforced only within
                      the event, via the same masked joint). ``None`` => unconditional.
      spike_sd, slab_sd: prior widths; default to ``0.03`` / ``0.5`` of the latent
                      site's domain span if ``None``.
      spike_w:        prior mass on the spike (how much you a-priori trust it).
      w_prior, w_couple: weights of the two terms (different scale from ``[0,1]``
                      probability residuals — tune them).

    Returns a ``(model, params) -> scalar`` closure. ``x``'s domain must cover the
    ``f``-values ``c`` can take, and the ``x``--``c`` bond needs rank >= 2 or the
    joint factorises and the latent goes inert. When ``given`` is set, put ``c`` and
    the event's variables near ``x`` so the conditional coupling stays cheap.
    """
    import numpy as np

    def reg(model, params):
        cen_x = np.asarray(model.disc.bin_centers(x_site), np.float64)
        cen_c = np.asarray(model.disc.bin_centers(c_site), np.float64)
        fx = jnp.asarray(cen_x if f is None else np.asarray(f(cen_x), np.float64), jnp.float32)
        span = float(model.disc.upper[c_site] - model.disc.lower[c_site])
        ssd = 0.03 * span if spike_sd is None else spike_sd
        wsd = 0.50 * span if slab_sd is None else slab_sd
        prior = jnp.asarray(spike_slab_prior(cen_c, target, ssd, wsd, spike_w))
        cc = jnp.asarray(cen_c, jnp.float32)

        # coupling table g(x, c): the joint conditioned on `given` (or plain joint).
        # Always normalised: the raw masked contraction carries a factor of Z, which
        # nothing pins, so an un-normalised coupling could be silently switched off
        # by rescaling the cores. Normalising the masked table yields p(x, c | given)
        # -- scale-free, and mass-weighting still makes slab bins costless.
        g = model.joint_marginal(params, (x_site, c_site), masks=given)
        if x_site > c_site:
            g = g.T                                             # orient to g[x, c]
        p_c_g = jnp.sum(g, axis=0)
        resid = (fx @ g) - cc * p_c_g                          # == p_c_g * (E[f(X)|c] - c)
        couple = jnp.sum(resid ** 2)

        # prior on the (unconditional) latent marginal p(c)
        p_c = model.joint_marginal(params, c_site)
        kl = jnp.sum(prior * (jnp.log(prior + _EPS) - jnp.log(p_c + _EPS)))
        return w_prior * kl + w_couple * couple

    return reg


def robust_mean(x_site, c_site, target, **kw):
    """Robust mean constraint ``E[X] ~ target`` — :func:`robust_expectation` with ``f=identity``."""
    return robust_expectation(x_site, c_site, target, f=None, **kw)


# ---------------------------------------------------------------------------
# belief-derived robust constraints (see chat derivation): a constraint is
# correct with prob 1-p_broken, in which case its value is N(target, value_sd);
# broken means it asserts NOTHING. Two faithful implementations:
#   mixture_expectation  -- scenario latent integrated out analytically (scalar)
#   belief_expectation   -- explicit latent with a BROKEN state (bin 0) + value
#                           bins, reverse-KL prior, hard within-branch coupling
# ---------------------------------------------------------------------------

def _functional_range(model, values):
    """(min, max) achievable value of the ``values`` functional (see ``expectation``)."""
    import numpy as np
    if isinstance(values, tuple):
        arr = np.asarray(values[1], np.float64)
        return float(arr.min()), float(arr.max())
    lo = sum(float(np.min(np.asarray(g, np.float64))) for g in values.values())
    hi = sum(float(np.max(np.asarray(g, np.float64))) for g in values.values())
    return lo, hi


def mixture_expectation(values, target, value_sd, *, p_broken=0.05, broken_density=None):
    """Exact robust expectation loss — the scenario latent integrated out.

    The NLL of the belief "with prob ``1-p_broken`` the constraint is correct and
    the true value of ``E[g(X)]`` is ``N(target, value_sd^2)``; with prob
    ``p_broken`` it asserts nothing":

        L(mu) = -log[ (1-p_broken) * N(mu; target, value_sd) + p_broken * h ]

    with ``mu = E[g(X)]`` (``values`` as in :meth:`TensorChain.expectation`) and
    ``h`` the density of a broken assertion (default: uniform over the achievable
    range of the functional). Quadratic with curvature ``1/value_sd^2`` near the
    target; plateaus (the constraint *surrenders*, gradient -> 0) beyond
    ``~value_sd * sqrt(2 log(ratio))``. **No tuning knobs**: ``p_broken`` and
    ``value_sd`` are the stated belief. Read the posterior trust with
    :func:`mixture_credence`.

    Caveat (inherent to any redescending loss): beyond the give-up radius it also
    exerts no *pull* — it cannot drag the model to a far target, only refine /
    release a near one. Use it for constraints you want robust, alongside firmer
    ones that shape the coarse fit.
    """
    import numpy as np

    def reg(model, params):
        mu = model.expectation(params, values)
        if broken_density is None:
            lo, hi = _functional_range(model, values)
            h = 1.0 / max(hi - lo, 1e-9)
        else:
            h = broken_density
        act = ((1.0 - p_broken) * jnp.exp(-0.5 * ((mu - target) / value_sd) ** 2)
               / (value_sd * float(np.sqrt(2.0 * np.pi))))
        return -jnp.log(act + p_broken * h + _EPS)

    return reg


def mixture_credence(model, params, values, target, value_sd, p_broken=0.05,
                     broken_density=None):
    """Posterior ``P(constraint correct)`` for a :func:`mixture_expectation` term.

    The mixture responsibility at the fitted ``mu = E[g(X)]``. Not a loss.
    """
    import numpy as np
    mu = float(model.expectation(params, values))
    if broken_density is None:
        lo, hi = _functional_range(model, values)
        h = 1.0 / max(hi - lo, 1e-9)
    else:
        h = broken_density
    act = ((1.0 - p_broken) * np.exp(-0.5 * ((mu - target) / value_sd) ** 2)
           / (value_sd * np.sqrt(2.0 * np.pi)))
    return float(act / (act + p_broken * h + _EPS))


def belief_expectation(x_site, c_site, target, value_sd, f=None, given=None, *,
                       p_broken=0.05, enforce=10.0):
    """Belief-derived robust constraint via a value latent with an explicit BROKEN state.

    ``c_site`` is a :func:`~calibrated_response.tn.belief_var`: **bin 0 = broken**
    ("the constraint asserts nothing" — no coupling, priced at exactly
    ``-log p_broken``), bins 1..K = the discretised *true value* ``v`` of
    ``E[f(X)[|given]]``, with pure-Gaussian prior ``N(target, value_sd)`` — no
    slab, so sliding the value by ``delta`` costs ``~delta^2/(2 value_sd^2)``
    nats. Give the var a span of about ``target +- (4-6) * value_sd``.

    The loss is the derived free energy

        KL( q(c) || p*(c) )   +   enforce * sum_k [ q_k * (E[f(X)|c=v_k] - v_k) ]^2

    with **reverse** KL (its minimiser is the Gibbs posterior
    ``q(c) proportional to p*(c) exp(-cost of living in scenario c)``) and the
    coupling enforced only on the value bins, mass-weighted and division-free.
    All *beliefs* live in ``(target, value_sd, p_broken)``; ``enforce`` is a
    numerical stiffness (the coupling is a definition, not a preference) — raise
    it until residuals are ~0, don't tune it as a belief. Read the posterior with
    :func:`belief_readout`; the model's predictive is the credence-weighted
    mixture over scenarios. ``given`` (an event ``{site: mask}``) makes the
    constraint conditional. Returns a ``(model, params) -> scalar`` closure.
    """
    import numpy as np

    def reg(model, params):
        d_c = model.dims[c_site]
        K = d_c - 1
        lo = float(model.disc.lower[c_site]); hi = float(model.disc.upper[c_site])
        v = lo + (np.arange(K) + 0.5) * (hi - lo) / K            # value-bin centres
        cen_x = np.asarray(model.disc.bin_centers(x_site), np.float64)
        fx = jnp.asarray(cen_x if f is None else np.asarray(f(cen_x), np.float64), jnp.float32)

        g = np.exp(-0.5 * ((v - target) / value_sd) ** 2)
        g = g / g.sum()
        pstar = np.concatenate([[p_broken], (1.0 - p_broken) * g]).astype(np.float32)
        kl = model.marginal_kl(params, c_site, pstar, direction="reverse")

        J = model.joint_marginal(params, (x_site, c_site), masks=given)  # scale-free
        if x_site > c_site:
            J = J.T                                              # orient to J[x, c]
        Jv = J[:, 1:]                                            # value bins only
        q_v = jnp.sum(Jv, axis=0)
        resid = (fx @ Jv) - jnp.asarray(v, jnp.float32) * q_v    # == q_k (E[f|v_k] - v_k)
        return kl + enforce * jnp.sum(resid ** 2)

    return reg


def belief_readout(model, params, c_site):
    """Diagnostics for a :func:`belief_expectation` latent — not a loss.

    Returns ``(p_broken, value_grid, value_posterior)``: the posterior probability
    the constraint is broken, the value-bin centres, and ``p(value | correct)``.
    """
    import numpy as np
    q = np.asarray(model.joint_marginal(params, c_site))
    K = len(q) - 1
    lo, hi = float(model.disc.lower[c_site]), float(model.disc.upper[c_site])
    v = lo + (np.arange(K) + 0.5) * (hi - lo) / K
    val = q[1:]
    return float(q[0]), v, val / (val.sum() + _EPS)


def onoff_expectation(x_site, c_site, target, value_sd, f=None, given=None, *,
                      p_broken=0.05):
    """Minimal robust constraint: a 2-state on/off latent gating a *marginal* penalty.

    ``c_site`` is a 2-state latent (``latent_var(name, 0., 1., 2)``): **state 0 =
    broken** ("the constraint asserts nothing"), state 1 = active. The loss is

        KL( q(c) || [p_broken, 1-p_broken] )
          + ( q_active * (E[f(X)[|given]] - target) )^2 / (2 value_sd^2)

    The distillation of :func:`belief_expectation` after two lessons. (1) Couple
    the **marginal**, not the conditional: ``E[f(X)|c]~c`` lets the model
    correlate ``X`` with each latent and satisfy every constraint within its own
    scenario slice while the predictive hedges; comparing marginal against
    target closes that escape (correlating ``X`` with ``c`` changes neither
    factor). (2) The value grid is redundant: the Gaussian belief
    ``N(target, value_sd)`` is already fully encoded in the quadratic weight
    ``1/(2 value_sd^2)``, so the K value bins just re-represent a posterior that
    stays at the target anyway.

    The ``q_active`` gate is division-free: convicting a constraint
    (``q_active -> 0``) smoothly switches its pull off at a price of exactly
    ``-log p_broken`` nats from the KL. Versus :func:`mixture_expectation`
    (which also prices the marginal): the mixture weight is a *model variable*
    the optimiser can slide along and back, not a redescending plateau it falls
    off — so no seed-dependent basin capture. All beliefs live in
    ``(target, value_sd, p_broken)``; there are no stiffness knobs.

    Caveats: a genuinely symmetric conflict (two equally-trusted, incompatible
    targets) is resolved by convicting one arbitrarily rather than hedging; and
    the latent carries no scenario structure — ``p(X | c)`` is not meaningful,
    only the marginal ``q`` (read trust as ``joint_marginal(params, c_site)[1]``).
    Returns a ``(model, params) -> scalar`` closure.
    """
    import numpy as np
    w = 1.0 / (2.0 * value_sd * value_sd)
    prior = jnp.asarray([p_broken, 1.0 - p_broken], jnp.float32)

    def reg(model, params):
        if given is None:
            cen = np.asarray(model.disc.bin_centers(x_site), np.float64)
            fx = jnp.asarray(cen if f is None else np.asarray(f(cen), np.float64),
                             jnp.float32)
            mu = model.expectation(params, {x_site: fx})
        else:
            mu = model.cond_expectation(params, x_site, given, f=f)
        q = model.joint_marginal(params, c_site)
        kl = jnp.sum(q * (jnp.log(q + _EPS) - jnp.log(prior)))
        return kl + w * (q[1] * (mu - target)) ** 2

    return reg


def projection_entropy(n_dirs=4, seed=0, n_grid=161):
    """Maximise the entropy of the projected distribution ``a·X`` over random ``a``.

    Unlike :func:`linear_isotropy` (which controls only the *variance* of ``a·X``),
    this uses the full projected pmf from :meth:`TensorChain._projection_pmf` and
    penalises ``-H(a·X)`` averaged over directions — so it fights *bunching*
    (multimodal / spiky projections) that a variance target cannot see. Returns a
    ``(model, params) -> scalar`` closure (minimise => maximise mean projected
    entropy). ``n_grid`` sets the value-grid resolution of the pmf.
    """
    import numpy as np

    def reg(model, params):
        dirs = np.random.default_rng(seed).normal(size=(n_dirs, model.n)).astype(np.float32)
        cores = model._cores(params)
        tot = 0.0
        for a in dirs:
            _, pmf = model._projection_pmf(cores, a, n_grid)
            tot = tot + jnp.sum(pmf * jnp.log(pmf + _EPS))       # = -H(a·X)
        return tot / n_dirs                                      # minimise => max entropy

    return reg


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

def combined_loss(model, constraints, regularizers=()):
    """Constraint SSE + sum of ``weight * reg(model, params)``.

    ``regularizers`` is a list of ``(fn_or_name, weight)``; names are looked up
    in :data:`REGULARIZERS`.
    """
    base = model.constraint_loss(constraints)     # SSE only
    regs = [(REGULARIZERS[f] if isinstance(f, str) else f, w) for f, w in regularizers]

    def loss(p):
        v = base(p)
        for fn, w in regs:
            v = v + w * fn(model, p)
        return v
    return loss


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def marginal_perplexity(model, params):
    """Mean effective #bins occupied per site, ``exp(H(p(X_i)))``.

    A spikiness gauge: near 1 => mass on one bin (degenerate); near ``n_bins``
    => spread. Not a loss, just for reporting.
    """
    import numpy as np
    vals = []
    for p in _site_marginals(model, params):
        p = np.asarray(p)
        H = -np.sum(p * np.log(p + _EPS))
        vals.append(float(np.exp(H)))
    return float(np.mean(vals))
