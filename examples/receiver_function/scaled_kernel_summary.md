# Summary: Reduced Likelihood with Kernel-Based Noise Estimation

## Overview

This work extends the `ReducedLikelihood` class in CoFI to support joint estimation of Earth model parameters and correlated noise properties (amplitude and correlation length) in receiver function inversion. The key contributions are:

1. A `SquaredExponentialKernel` class for parameterizing noise correlation
2. Support for gradient-free optimization across all `ReducedLikelihood` cases
3. A new `kernel_full` case with explicit noise amplitude
4. A two-step inversion workflow that separates noise estimation from model fitting

---

## 1. SquaredExponentialKernel

**File**: `cofi-dev/src/cofi/utils/_kernel.py`

A squared exponential (Gaussian/RBF) kernel parameterized by log-correlation-length `eta = log(l)`:

$$K_{ij}(\eta) = \exp\left(-\frac{(x_i - x_j)^2}{2\ell^2}\right) + \delta_{ij}\epsilon$$

where `epsilon` is a small nugget for numerical stability. The class provides:

- `evaluate(eta)` — kernel matrix `K`
- `evaluate_derivative(eta)` — derivative `dK/deta`
- `n_params` — number of hyperparameters (1)

---

## 2. Gradient-Free Optimization Support

**File**: `cofi-dev/src/cofi/utils/_reduced_likelihood.py`

### Problem

The original `ReducedLikelihood` required a Jacobian matrix `G` for all evaluations, making it incompatible with derivative-free optimizers (e.g., Nelder-Mead).

### Solution

Split the monolithic `_evaluate()` method into two tiers:

- **`_evaluate_loglik(model)`** — computes log-likelihood, requires only forward function (no `G`). Caches intermediates (`L`, `alpha`, `a`, etc.) for reuse by derivatives.
- **`_evaluate_derivatives(model)`** — computes gradient and Hessian, requires `G`. Uses cached intermediates from `_evaluate_loglik`.

Additional changes:

- Added `n_params` constructor parameter to specify model dimensions when `G=None`
- Added `G` property with setter that invalidates the internal cache
- Updated `_validate_model` to skip size checks when `model_shape` is `None` (neither `G` nor `n_params` provided)
- `log_likelihood()` and `get_ml_cov()` work without `G`; `gradient()` and `hessian()` raise `ValueError` if `G` is not set

This applies to **all** cases: `none`, `scaled`, `kernel`, `kernel_full`, `spherical`, `diag`, `full`.

---

## 3. `kernel` Case (Profile Likelihood)

Profiles out the noise amplitude `sigma_d` analytically, estimating only `eta = log(l)` alongside the physical model `m`.

**Model vector**: `[m_phys (n_m), eta (1)]`

**Log-likelihood**:

$$\ell(m, \eta) = -\frac{1}{2}\left[\log|K(\eta)| + N \log(r^T K^{-1}(\eta)\, r)\right]$$

where `r = d_obs - f(m)` is the residual and `N` is the number of data points.

**ML covariance**: `Cd_ml = (a/N) * K(eta)` where `a = r^T K^{-1} r`.

---

## 4. `kernel_full` Case (Explicit Sigma)

**File**: `cofi-dev/src/cofi/utils/_reduced_likelihood.py`

Keeps `sigma_d` as an explicit parameter via `phi = log(sigma_d)`, allowing it to be constrained by priors.

**Model vector**: `[m_phys (n_m), phi (1), eta (1)]`

**Covariance**: `C_d = exp(2*phi) * K(eta)`

**Log-likelihood**:

$$\ell(m, \phi, \eta) = -\frac{1}{2}\left[2N\phi + \log|K(\eta)| + e^{-2\phi}\, r^T K^{-1}(\eta)\, r\right]$$

**Gradient** `[grad_m, grad_phi, grad_eta]`:

$$\nabla_m = e^{-2\phi}\, G^T \alpha, \quad \nabla_\phi = -N + e^{-2\phi}\, a, \quad \nabla_\eta = -\frac{1}{2}\left[\text{tr}(K^{-1} K_\eta) - e^{-2\phi}\, \alpha^T K_\eta \alpha\right]$$

where `alpha = K^{-1} r`, `a = r^T alpha`, and `K_eta = dK/deta`.

**Hessian**: Block structure over `[m, phi, eta]` with cross-terms. Sign-critical terms:

- `H_m_eta = -exp(-2*phi) * G^T K^{-1} K_eta alpha`
- `H_phi_eta = -exp(-2*phi) * alpha^T K_eta alpha`

---

## 5. Two-Step Inversion Workflow

**File**: `cofi-examples/examples/receiver_function/receiver_function_kernel.ipynb`

### Motivation

Estimating noise parameters from the entire dataset contaminates them with model misfit. A pre-signal window where the forward model response is zero contains only noise.

### Step 1: Noise-Window Estimation

- Select pre-signal window: `t in [-5, -0.5] s`
- Use `ReducedLikelihood` with `case='kernel'` and `n_params=0` (no physical model)
- Forward function returns zeros (no signal in noise window)
- Optimize `eta` using `scipy.optimize.minimize_scalar`
- Extract `sigma_d_hat` from `get_ml_cov()`: since `K[0,0] = 1`, `Cd_ml[0,0] = sigma_d^2`
- Compute `phi_hat = log(sigma_d_hat)`

### Step 2: Joint Inversion with Priors

- Use `ReducedLikelihood` with `case='kernel_full'` over the full dataset
- Model vector: `[m (8), phi (1), eta (1)]` = 10 parameters
- Gaussian priors via `QuadraticReg` with diagonal weighting matrix `W = diag(1/sigma_prior)`:
  - Weak prior on physical model (`sigma = 10`)
  - Moderate prior on `phi` and `eta` (`sigma = 0.5`), centered on noise-window estimates
- Objective: `-log_likelihood + 0.5 * ||W(x - x_ref)||_2^2`
- Optimize with Nelder-Mead

---

## 6. Manager Updates

**File**: `cofi-dev/src/cofi/utils/_reduced_likelihood_manager.py`

The `ReducedLikelihoodManager` was updated to support `kernel` and `kernel_full` cases, handling:

- Kernel hyperparameter slicing from the model vector
- Mixed cases (some datasets with kernels, some without)
- Jacobian caching across multiple datasets

---

## 7. Tests

**File**: `cofi-dev/tests/cofi_utils/test_reduced_likelihood.py`

49 tests covering:

- `G=None` support for all cases (log-likelihood works, gradient/hessian raise)
- Dynamic `G` assignment and cache invalidation
- `n_params` consistency validation
- `kernel_full`: shapes, numerical gradient, numerical Hessian
- `kernel_full` without `G`
- Multiple `phi`/`eta` values for `kernel_full`

---

## Files Modified

| File | Description |
|------|-------------|
| `cofi-dev/src/cofi/utils/_kernel.py` | New `SquaredExponentialKernel` class |
| `cofi-dev/src/cofi/utils/_reduced_likelihood.py` | Two-tier evaluation, `n_params`, `G` property, `kernel_full` case |
| `cofi-dev/src/cofi/utils/_reduced_likelihood_manager.py` | Kernel support, model slicing |
| `cofi-dev/src/cofi/utils/__init__.py` | Exports for new classes |
| `cofi-dev/tests/cofi_utils/test_reduced_likelihood.py` | 49 tests including all new functionality |
| `cofi-examples/.../receiver_function_kernel.ipynb` | Two-step workflow notebook |
