# Gaussian VI Implementation Notes

## Overview

The `researcher_vi.py` notebook extends the nonlinear surface-wave tomography workflow
with Gaussian variational inference (VI) via `cofi.gaussian_vi`. The solver approximates
the posterior distribution over the slowness field with a Gaussian, parameterised by a
mean vector and a sparse precision matrix, optimised by maximising the evidence lower
bound (ELBO).

The solver has three phases:

1. **MAP initialisation** — L-BFGS-B to find the maximum a posteriori estimate
2. **Gaussian VI** — natural-gradient updates on the mean and precision of the
   Gaussian approximation, tracked via the ELBO
3. **Optional normalising flow** — sinh-arcsinh transform for non-Gaussian posteriors

All internal linear algebra uses `sksparse.cholmod` (CHOLMOD) for sparse Cholesky
factorisation — no dense conversion occurs.

---

## Prior specification

The prior precision is built from the SPDE Matérn factor:

```
Q_prior = R.T @ R
```

where `R = τ √(h_x h_y) (κ²I − L_h)` is the sparse factor from `cofi.utils.SPDEMaternReg`,
with parameters:

| Symbol | Expression | Value | Meaning |
|--------|-----------|-------|---------|
| `ell`  | `√2 × 5 × 0.3°` | 2.12° | Matérn length scale |
| `sigma` | `5 × 0.02 / (2√π)` | 0.028 s/km | Prior marginal std dev |
| `κ` | `√2 / ell` | 0.667 | Inverse length scale |
| `τ` | `1 / (2√π κ σ)` | ~15.0 | Precision scaling |

The marginal standard deviation of the field under `Q_prior` is exactly `sigma` by
construction — that is how τ is derived.

---

## Prior scaling issue

### The problem

At `sigma = 0.028 s/km` and a background velocity of 3 km/s, the prior allows:

- **1σ**: ±0.028 s/km → ±8.5% velocity variation
- **2σ**: ±0.056 s/km → ±17% velocity variation

The nonlinear inversion (and the VI MAP phase) produce velocities in [2.13, 3.59] km/s.
Converting to slowness deviations from the 1/3.0 s/km background:

| Velocity | Slowness | Deviation from m₀ | In units of σ |
|----------|----------|--------------------|---------------|
| 2.13 km/s | 0.469 s/km | +0.136 s/km | **+4.9σ** |
| 3.59 km/s | 0.279 s/km | −0.055 s/km | **−2.0σ** |

The data requires perturbations up to ~5σ from the prior mean. Under the Gaussian prior
this is extremely improbable (p ≈ 10⁻⁶), so the prior strongly resists these deviations.

### Impact on VI

The MAP phase (Phase 1) is largely unaffected — L-BFGS-B can push through the prior
penalty when the data likelihood gradient dominates. The MAP velocity range
[2.13, 3.59] km/s is essentially identical to the standalone nonlinear inversion
[2.15, 3.56] km/s.

However, in the VI phase (Phase 2), the tight prior cripples the **uncertainty estimates**.
The ELBO oscillates without converging (~±200 around −90,000 over 10 iterations) because
the natural-gradient updates cannot reconcile the wide MAP solution with the narrow prior:
the posterior precision gets pulled between the data (which wants large variance to
accommodate the MAP) and the prior (which insists on small variance).

### Relationship to mu_nonlin

In the nonlinear least-squares formulation, the augmented system applies `√mu_nonlin × R`,
making the effective prior precision `mu_nonlin × R.T @ R` with effective sigma:

```
σ_eff = σ / √mu_nonlin
```

| Mode | mu_nonlin | σ_eff (s/km) | ±2σ velocity range |
|------|-----------|-------------|-------------------|
| Fast (1% rays) | 0.024 | 0.18 | ±55% — wide, data-dominated |
| Full (100% rays) | 2.0 | 0.020 | ±6% — very tight |

The VI notebook passes `Q_prior = R.T @ R` without mu_nonlin, corresponding to
`mu_nonlin = 1` and `σ_eff = 0.028`. This is tighter than the fast-mode calibration
but looser than full-mode.

### Resolution

The prior sigma should encode a genuine physical belief about plausible velocity
perturbations, independent of the regularisation weight. A reasonable sigma would place
the observed velocity range within ~2σ:

```
σ_phys ≈ max_deviation / 2 ≈ 0.136 / 2 ≈ 0.07 s/km
```

This corresponds to ±20% velocity variation at 2σ — consistent with the range of
Rayleigh-wave phase velocities observed across continental Australia at 5 s period.

If widening sigma causes underfitting (solution too rough or noisy), the correct
adjustment is to tighten C_d (reduce the assumed data noise σ_d), which increases the
data's relative weight. The data-vs-prior balance should be controlled through C_d,
not by narrowing the physical prior.

---

## Current hyperparameters

```python
inv_options.set_params(
    prior_precision=Q_prior,        # R.T @ R, no mu_nonlin scaling
    prior_mean=m0_pyfm2d,           # uniform 1/3.0 s/km
    num_iterations=10,
    num_samples=1,                  # single-sample ELBO gradient (noisy)
    map_num_iterations=10,
    learning_rate_mean=0.02,
    learning_rate_precision=0.05,
    random_seed=42,
)
```

The single-sample gradient (`num_samples=1`) also contributes to ELBO oscillation.
Increasing to 4–10 samples would reduce gradient variance, but the fundamental issue
is the prior scaling.

---

## Dependencies

- `cofi.gaussian_vi` solver
- `sksparse.cholmod` (CHOLMOD) — hard dependency for sparse Cholesky
- `cofi.utils.SPDEMaternReg` — builds the sparse Matérn factor R
- Virtual environment: `~/.virtualenvs/inlab`
