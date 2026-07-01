# `InversionOptions` Changes Across CoFI Example Notebooks

The core CoFI claim is that switching algorithms requires only updating `InversionOptions` — the `BaseProblem` and `Inversion` objects remain unchanged. This document summarises every distinct `InversionOptions` configuration used across the three example notebooks.

---

## `educator.py`

The same `inv_options` object is created once and progressively reconfigured across a polynomial-fitting example that steps through linear, nonlinear, and Bayesian approaches.

| Stage | `set_solving_method` | `set_tool` | Key `set_params` args |
|---|---|---|---|
| 1 | `'matrix solvers'` | *(default: `scipy.linalg.lstsq`)* | none |
| 2 | `'optimization'` | `scipy.optimize.minimize` | `method="Nelder-Mead"` |
| 3 | `'sampling'` | `emcee` | `nwalkers`, `nsteps`, `initial_state`, `progress=True` |
| 4 | `'sampling'` | `bayesbay` | `log_like_ratio_func`, `perturbation_funcs`, `walkers_starting_states`, `n_chains`, `n_iterations`, `burnin_iterations`, `save_every`, `verbose` |

**Notes:**
- Transitions 1→2, 2→3, and 3→4 each call `inv_options.hyper_params.clear()` to flush stale solver parameters before setting new ones.
- Stage 3 (emcee) is reused identically in three sub-examples (polynomial, sea-level 3-station, sea-level 10-station); only `initial_state` differs between them.
- `set_solving_method` is set explicitly at every stage.

---

## `practitioner.py`

The notebook addresses seismic travel-time tomography — a larger-scale problem with a sparse linear system, nonlinear inversion, and trans-dimensional Bayesian sampling. The same `inv_options` object is reconfigured throughout.

| Stage | `set_solving_method` | `set_tool` | Key `set_params` args |
|---|---|---|---|
| 1 | *(not set)* | `scipy.sparse.linalg` | `algorithm="minres"` |
| 2 | *(not set)* | `scipy.optimize.minimize` | `method="L-BFGS-B"`, `callback`, `options={maxiter=10, ftol=1e-6, gtol=1e-5}` |
| 3 | *(not set)* | `scipy.optimize.least_squares` | `method="trf"`, `max_nfev=10`, `verbose=2`, `x_scale`, `ftol=1e-4`, `xtol=1e-4`, `tr_solver="lsmr"` |
| 4 | `'sampling'` | `bayesbay` | `walkers_starting_states`, `perturbation_funcs`, `log_like_ratio_func`, `n_chains`, `n_iterations`, `burnin_iterations`, `save_every`, `verbose=False` |

**Notes:**
- `set_solving_method` is omitted for stages 1–3; CoFI infers the method from the tool name. It is set explicitly only when switching to `"sampling"` for BayesBay.
- `hyper_params.clear()` is called before stages 2 and 3.
- Stage 3 (`least_squares`) uses `x_scale` (a fraction of the starting model `m0`) to limit the trust-region step size — a practitioner-level concern not present in `educator.py`.
- Unlike `educator.py`, emcee is not used; the notebook goes directly to BayesBay for the Bayesian stage.

---

## `developer.py`

This notebook demonstrates how to register a custom solver (the Slime Mould Algorithm from the `mealpy` library) as a CoFI backend. Rather than reusing one `inv_options` object, each demonstration creates a fresh instance to isolate tests of the plugin interface.

| Stage | `set_solving_method` | `set_tool` | Key `set_params` args |
|---|---|---|---|
| 1 | `'optimization'` | `scipy.optimize.minimize` | *(none — comparison baseline only)* |
| 2 | `'optimization'` | `mealpy.sma.demo.from.this.notebook` | `epoch=200`, `pop_size=50`, `seed=42` |
| 3 | `'optimization'` | `mealpy.sma.demo.from.this.notebook` | `epoch` and `pop_size` vary per run; `seed` varies for reproducibility tests |
| 4 | `'optimization'` | *(loop over registered tool names)* | `**params` dispatched generically from a dict |

**Notes:**
- The tool name `"mealpy.sma.demo.from.this.notebook"` is a dynamically registered plugin; all other CoFI machinery is identical to built-in solvers.
- `set_solving_method` is always set explicitly (`'optimization'`), unlike `practitioner.py`.
- Only the `optimization` solving method is used; no sampling or linear solvers appear.
- The notebook also contains a brief `scipy.optimize.minimize` block (stage 1) purely to establish a deterministic baseline for comparison.

---

## Cross-notebook comparison

| Feature | `educator.py` | `practitioner.py` | `developer.py` |
|---|---|---|---|
| Reuses one `inv_options` object | Yes | Yes | No — fresh per test |
| Calls `hyper_params.clear()` between stages | Yes | Yes | N/A |
| Sets `set_solving_method` explicitly | Always | Only for `"sampling"` | Always |
| Solving methods used | matrix solvers, optimisation, sampling | *(inferred)* sparse linear, optimisation, sampling | optimisation only |
| Linear solver | `scipy.linalg.lstsq` | `scipy.sparse.linalg` (minres) | none |
| Optimisation solver(s) | `scipy.optimize.minimize` (Nelder-Mead) | `scipy.optimize.minimize` (L-BFGS-B), `scipy.optimize.least_squares` (trf) | `scipy.optimize.minimize`, custom `mealpy` SMA |
| Sampling solver(s) | emcee, then bayesbay | bayesbay only | none |
| Custom/plugin solver | No | No | Yes (`mealpy` SMA registered as CoFI backend) |

---

## What never changes

Across all three notebooks and all algorithm switches, the following CoFI calls are never modified:

- `BaseProblem` — the problem definition (data, forward function, priors, etc.) is set up once and only augmented, never rebuilt from scratch.
- `Inversion(inv_problem, inv_options)` — the same constructor call is used each time, regardless of solver.
- The pattern `Inversion(...).run()` — solver dispatch is fully internal to CoFI.
