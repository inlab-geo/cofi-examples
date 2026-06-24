# Proposed Updates to `cofi.utils` Regularization

This note describes concrete code changes required to (1) fix sparse matrix
handling throughout `cofi.utils`, and (2) add a new `SPDEMaternReg` utility class
implementing Matérn ν=1 regularization for 2D spatial fields via the SPDE sparse
precision approach. All changes are to files in `cofi/utils/`.

---

## Change 1 — Fix sparse matrix combination in `LpNormRegularization`

**File:** `cofi/utils/_reg_lp_norm.py`
**Method:** `_generate_weighting_matrix`

### Problem

For a 2D `model_shape` with `weighting_matrix` in
`{"flattening", "roughening", "smoothing"}`, the code generates two sparse
directional difference operators via `findiff`, then immediately converts them
to dense arrays before stacking:

```python
# Current (lines ~203–207) — WRONG
matx = d_dx.matrix((nx, ny))   # scipy sparse
maty = d_dy.matrix((nx, ny))   # scipy sparse
self._weighting_matrix = np.vstack(
    (matx.toarray(), maty.toarray())   # converts to dense → ~6 GB for 150×130 grid
)
```

### Fix

Replace `np.vstack` with `scipy.sparse.vstack` and drop the `.toarray()` calls:

```python
# Fixed
from scipy import sparse          # already imported elsewhere in the file
self._weighting_matrix = sparse.vstack([matx, maty], format="csr")
```

This is a one-line change. The resulting matrix is a sparse CSR matrix of shape
`(2·nx·ny, nx·ny)` — consistent with the 1D case, which already returns a sparse
matrix — and reduces memory from ~6 GB to ~10 MB for a 150×130 grid.

---

## Change 2 — Sparse-safe Hessian in `LpNormRegularization`

**File:** `cofi/utils/_reg_lp_norm.py`
**Method:** `hessian`

### Problem

The Hessian computation builds an intermediate dense diagonal matrix with `np.diag`:

```python
# Current (line ~154) — creates dense n×n intermediate
def hessian(self, model):
    W = self._weighting_matrix
    ...
    hess_lp_norm = self._lp_norm_hessian(weighted_diff_m)  # vector of length n_rows
    return W.T @ np.diag(hess_lp_norm) @ W
```

When `W` is sparse (as after Fix 1), `np.diag(hess_lp_norm)` creates a dense
`n_rows × n_rows` matrix, immediately destroying the sparsity benefit and
potentially causing memory errors.

### Fix

Replace `np.diag` with `scipy.sparse.diags`:

```python
# Fixed
def hessian(self, model):
    W = self._weighting_matrix
    flat_m = self._validate_model(model)
    diff_m = self._model_diff_to_ref(flat_m)
    weighted_diff_m = W @ diff_m
    hess_lp_norm = self._lp_norm_hessian(weighted_diff_m)
    if sparse.issparse(W):
        return W.T @ sparse.diags(hess_lp_norm) @ W
    else:
        return W.T @ np.diag(hess_lp_norm) @ W
```

For the common case `p=2` (`QuadraticReg`), `hess_lp_norm = 2 · ones`, so
`sparse.diags(hess_lp_norm) = 2I` and the result simplifies to `2 W.T W` — a
sparse matrix with the same sparsity pattern as `W.T W`.

---

## Change 3 — `GaussianPrior` accept sparse `C_m_inv`

**File:** `cofi/utils/_reg_model_cov.py`
**Method:** `_prepare_covariance_matrix_inv`

### Problem

The type check on `model_covariance_inv` accepts only `np.ndarray` or `tuple`,
raising `TypeError` for any scipy sparse matrix:

```python
# Current (lines ~98–114)
if isinstance(model_covariance_inv, np.ndarray):
    ...
elif isinstance(model_covariance_inv, (tuple, list)):
    ...
else:
    raise TypeError(...)   # sparse matrix hits this branch
```

### Fix

Extend the `np.ndarray` branch to also accept sparse matrices, add `from scipy
import sparse` at the top of the file, and wrap the shape-check to handle both:

```python
from scipy import sparse   # add to imports

def _prepare_covariance_matrix_inv(self, model_covariance_inv, mean_model):
    if isinstance(model_covariance_inv, np.ndarray) or sparse.issparse(model_covariance_inv):
        mu = self._mu
        Cminv = model_covariance_inv
        if Cminv.shape != (mu.shape[0], mu.shape[0]):
            raise ValueError(
                f"({(mu.shape[0], mu.shape[0])}) expected for the shape of "
                f"model_covariance_inv but got matrix of shape {Cminv.shape}"
            )
        self._Cminv = Cminv
    elif isinstance(model_covariance_inv, (tuple, list)):
        self._Cminv = self._generate_covariance_matrix_inv(
            mean_model.shape,
            model_covariance_inv[0],
            model_covariance_inv[1],
        )
    else:
        raise TypeError(
            "numpy.ndarray, scipy sparse matrix, or (tuple, float) expected "
            f"for `model_covariance_inv` but got {type(model_covariance_inv)}"
        )
```

### Notes

- `reg` (`diff_m.T @ self._Cminv @ diff_m`) and `gradient`
  (`2 * self._Cminv @ diff_m`) work unchanged for sparse `_Cminv`.
- `hessian` returns `2 * self._Cminv`, which will now be sparse when a sparse
  matrix is passed. Callers that expect a dense array (e.g. direct matrix
  inversion) should call `.toarray()` themselves.
- `_generate_covariance_matrix_inv` still builds a **dense** inverse via
  `np.linalg.inv` — this is a separate limitation described in
  `matern_prior_theory.md`. The SPDE approach in Change 4 below is the
  recommended alternative for large 2D grids.

---

## Change 4 — New `SPDEMaternReg` class

**New file:** `cofi/utils/_reg_matern.py`

### Purpose

A drop-in alternate for `QuadraticReg` with a bring-your-own matrix, providing
a user-facing API for Matérn ν=1 regularization on 2D grids. The precision factor
`R = \tau M^{1/2} (κ²I − L_h)` is built automatically as a sparse matrix from
physically interpretable parameters.

### Implementation

```python
from numbers import Number
from typing import Optional
import numpy as np
from scipy import sparse

from ._reg_lp_norm import QuadraticReg


class SPDEMaternReg(QuadraticReg):
    r"""Sparse Matérn ν=1 regularization for 2D spatial fields via the SPDE approach.

    Implements the regularization term

    .. math::

        \mathcal{R}(\mathbf{m}) = \|\mathbf{R}(\mathbf{m} - \mathbf{m}_0)\|^2,
        \quad \mathbf{R} = \tau \mathbf{M}^{1/2}(\kappa^2 \mathbf{I} - \mathbf{L}_h),
        \quad \kappa = \sqrt{8} / \rho,
        \quad \tau = \frac{1}{2 \sqrt{\pi}\,\kappa\,\sigma}

    where :math:`\mathbf{L}_h` is the grid-spacing-aware 2D discrete Laplacian and
    :math:`\mathbf{M}` is the lumped mass matrix. For a uniform grid,
    :math:`\mathbf{M} \approx h_x h_y \mathbf{I}`, so
    :math:`\mathbf{R} = \tau \sqrt{h_x h_y}(\kappa^2 \mathbf{I} - \mathbf{L}_h)`.
    The corresponding precision is
    :math:`Q = \mathbf{R}^\top \mathbf{R} = \tau^2 B_h^\top M B_h`, with
    :math:`B_h = \kappa^2 I - \mathbf{L}_h`. On a unit grid this reduces to the
    shorthand :math:`Q = \tau^2(\kappa^2\mathbf{I} - \mathbf{L})^2`. The matrix
    :math:`\mathbf{R}` is sparse (:math:`O(n)` non-zeros), unlike the dense
    precision matrix of a standard Gaussian prior.

    Parameters
    ----------
    model_shape : tuple of (int, int)
        Shape of the 2D model grid ``(n_lon, n_lat)`` (or equivalently any two
        positive integers whose product is the number of model parameters).
    rho : float
        Matérn practical range in the same physical units as ``grid_spacing``.
    sigma : float, optional
        Prior marginal standard deviation of model perturbations.
    grid_spacing : float or tuple of (float, float), optional
        Physical spacing of the regular grid in each coordinate direction.
    reference_model : np.ndarray, optional
        Background model :math:`\mathbf{m}_0`. If provided, the penalty is on
        :math:`\mathbf{m} - \mathbf{m}_0`; if omitted, the penalty is on
        :math:`\mathbf{m}` directly.

    Examples
    --------
    >>> from cofi.utils import SPDEMaternReg
    >>> import numpy as np
    >>> reg = SPDEMaternReg(model_shape=(10, 8), rho=4.0, sigma=0.02)
    >>> reg(np.zeros((10, 8)))
    0.0
    """

    def __init__(
        self,
        model_shape: tuple,
        rho: float,
        sigma: float = 1.0,
        grid_spacing=1.0,
        reference_model: Optional[np.ndarray] = None,
    ):
        if len(model_shape) != 2:
            raise ValueError(
                f"SPDEMaternReg requires a 2D model_shape (n_lon, n_lat), "
                f"got {model_shape}"
            )
        n_lon, n_lat = model_shape
        n_params = n_lon * n_lat
        h_lon, h_lat = _normalize_grid_spacing(grid_spacing)
        kappa = np.sqrt(8.0) / rho
        kappa2 = kappa ** 2
        tau = 1.0 / (2.0 * np.sqrt(np.pi) * kappa * sigma)

        # 1D tridiagonal Laplacian operators (full square, not truncated)
        L1d_lat = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n_lat, n_lat), format="csr")
        L1d_lon = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n_lon, n_lon), format="csr")

        # 2D Laplacian via Kronecker products
        L_full = (
            sparse.kron(sparse.eye(n_lon), L1d_lat / h_lat**2, format="csr")
            + sparse.kron(L1d_lon / h_lon**2, sparse.eye(n_lat), format="csr")
        )

        # Sparse precision factor R = tau * M^{1/2} * (κ²I − L_h)
        R = (
            tau
            * np.sqrt(h_lon * h_lat)
            * (kappa2 * sparse.eye(n_params, format="csr") - L_full)
        )

        # Flatten reference model if provided
        ref = np.ravel(reference_model) if reference_model is not None else None

        super().__init__(
            weighting_matrix=R,
            model_shape=(n_params,),
            reference_model=ref,
        )

        # Store parameters for inspection
        self._L_corr = L_corr
        self._sigma = sigma
        self._kappa2 = kappa2
        self._2d_shape = model_shape

    @property
    def L_corr(self) -> float:
        """Correlation length in grid cells."""
        return self._L_corr

    @property
    def sigma(self) -> float:
        """Prior marginal standard deviation."""
        return self._sigma

    @property
    def kappa(self) -> float:
        """Wavenumber parameter κ = 1/L_corr."""
        return 1.0 / self._L_corr

    @property
    def grid_shape(self) -> tuple:
        """Original 2D grid shape (n_lon, n_lat)."""
        return self._2d_shape
```

### Design notes

- Inherits all `reg`, `gradient`, and `hessian` methods from `QuadraticReg`
  (and thereby `LpNormRegularization`), which already implement the
  `||R(m − m₀)||²` formulation correctly once `reference_model` is set.
- After Fixes 1 and 2, the inherited `hessian` returns a **sparse** matrix
  (`2 R.T R`), enabling sparse Hessian assembly throughout.
- `R` is built as CSR format for efficient row slicing (row operations dominate
  in sparse matrix-vector products).
- The `sigma` parameter scales the entire operator, so `mu` (the regularization
  weight passed externally) acts as a dimensionless multiplier on top of the
  physically-scaled prior — see `matern_prior_theory.md` for details.

---

## Change 5 — Update module exports

**File:** `cofi/utils/__init__.py`

Add the new class to imports, `__all__`, and the inheritance diagram:

```python
r"""Utility classes and functions (e.g. to generate regularization terms and more)

The class inheritance of regularization classes:

.. mermaid::

    graph TD;
    BaseRegularization --> LpNormRegularization;
    LpNormRegularization --> QuadraticReg;
    LpNormRegularization --> SPDEMaternReg;
    BaseRegularization --> ModelCovariance;
    ModelCovariance --> GaussianPrior;

"""

from ._reg_base import BaseRegularization
from ._reg_lp_norm import LpNormRegularization, QuadraticReg
from ._reg_model_cov import ModelCovariance, GaussianPrior
from ._reg_matern import SPDEMaternReg          # new

from ._multiple_runs import InversionPool


__all__ = [
    "BaseRegularization",
    "LpNormRegularization",
    "QuadraticReg",
    "SPDEMaternReg",                            # new
    "ModelCovariance",
    "GaussianPrior",
    "InversionPool",
]
```

---

## Summary of changes

| # | File | Change | Impact |
|---|------|--------|--------|
| 1 | `_reg_lp_norm.py` | `scipy.sparse.vstack` instead of `np.vstack(…toarray())` | 2D findiff operators stay sparse |
| 2 | `_reg_lp_norm.py` | `sparse.diags` instead of `np.diag` in `hessian` | Hessian stays sparse when W is sparse |
| 3 | `_reg_model_cov.py` | Accept sparse `C_m_inv` in `GaussianPrior` | Allows user-supplied sparse precision matrices |
| 4 | `_reg_matern.py` (new) | `SPDEMaternReg` class | Matérn ν=1 prior, sparse, 2D, user-friendly API |
| 5 | `__init__.py` | Export `SPDEMaternReg` | Available as `cofi.utils.SPDEMaternReg` |

Changes 1 and 2 are bug fixes with no API changes. Change 3 is a backwards-compatible
extension. Changes 4 and 5 are additive. All changes preserve existing behaviour for
1D models and bring-your-own matrices.
