# Parameter Mapping: Old (`sparsefix`) ↔ New (`sparsefix-matern`) SPDE Matérn Formulation

## Overview

The `sparsefix` branch of CoFI implemented the SPDE Matérn ν=1 precision factor with a
simplified parameterisation (no grid spacing, no SPDE-consistent τ scaling). The
`sparsefix-matern` branch corrects this to the standard SPDE parameterisation from
Lindgren, Rue & Lindström (2011). This note derives the exact parameter mapping so that
both formulations produce the same precision matrix $Q$.

---

## Old formulation (`sparsefix`)

The old branch defines the precision factor as

$$
\mathbf{R}_{\text{old}} = \frac{1}{\sigma_{\text{old}}} \left( \kappa_{\text{old}}^2 \mathbf{I} - \mathbf{L} \right),
\qquad \kappa_{\text{old}} = \frac{1}{L_{\text{corr}}},
$$

where $\mathbf{L}$ is the 2D discrete Laplacian on a unit grid (no $1/h^2$ scaling).
The precision matrix is

$$
Q_{\text{old}} = \mathbf{R}_{\text{old}}^\top \mathbf{R}_{\text{old}}
= \frac{1}{\sigma_{\text{old}}^2}
  \left( \kappa_{\text{old}}^2 \mathbf{I} - \mathbf{L} \right)^2.
$$

Here $L_{\text{corr}}$ is measured in **grid cells** and $\sigma_{\text{old}}$ is a raw
scaling parameter (not the marginal standard deviation of the Matérn field).

---

## New formulation (`sparsefix-matern`)

The new branch implements the SPDE-consistent parameterisation with physical grid spacing:

$$
\mathbf{R}_{\text{new}} = \tau \sqrt{h_x h_y}
  \left( \kappa_{\text{new}}^2 \mathbf{I} - \mathbf{L}_h \right),
\qquad
\kappa_{\text{new}} = \frac{\sqrt{2}}{\ell},
\qquad
\tau = \frac{1}{2\sqrt{\pi}\,\kappa_{\text{new}}\,\sigma_{\text{new}}},
$$

where $\mathbf{L}_h$ is the grid-spacing-aware Laplacian (with $1/h^2$ scaling applied
to each directional component), $\ell$ is the Matérn length scale in **physical units**
(same units as $h_x$, $h_y$), and $\sigma_{\text{new}}$ is the marginal standard deviation
of the continuous Matérn field. The precision matrix is

$$
Q_{\text{new}} = \mathbf{R}_{\text{new}}^\top \mathbf{R}_{\text{new}}
= \tau^2 h_x h_y
  \left( \kappa_{\text{new}}^2 \mathbf{I} - \mathbf{L}_h \right)^2.
$$

---

## Relating the two Laplacians

On a uniform grid with isotropic spacing $h_x = h_y = h$, the grid-spacing-aware
Laplacian is

$$
\mathbf{L}_h = \frac{\mathbf{L}}{h^2},
$$

where $\mathbf{L}$ is the unit-grid (tridiagonal) Laplacian used in the old code. This is
because the standard second-order finite-difference stencil for $\partial^2/\partial x^2$
divides by $h^2$.

---

## Derivation of the mapping

Setting $Q_{\text{old}} = Q_{\text{new}}$ requires matching both the **operator** and the
**scalar prefactor**.

### Step 1: Match the operator

The operator in $Q_{\text{old}}$ is $(\kappa_{\text{old}}^2 \mathbf{I} - \mathbf{L})^2$.
The operator in $Q_{\text{new}}$ involves $\mathbf{L}_h = \mathbf{L}/h^2$:

$$
\kappa_{\text{new}}^2 \mathbf{I} - \mathbf{L}_h
= \kappa_{\text{new}}^2 \mathbf{I} - \frac{\mathbf{L}}{h^2}
= \frac{1}{h^2}\left(\kappa_{\text{new}}^2 h^2 \mathbf{I} - \mathbf{L}\right).
$$

For the operators to match we need

$$
\kappa_{\text{old}}^2 = \kappa_{\text{new}}^2 h^2.
$$

Substituting the definitions:

$$
\frac{1}{L_{\text{corr}}^2} = \frac{2}{\ell^2} \cdot h^2,
$$

$$
\boxed{\ell = \sqrt{2}\, L_{\text{corr}}\, h.}
$$

This makes physical sense: $L_{\text{corr}}$ is in grid cells, so $L_{\text{corr}} \cdot h$
converts to physical units, and the $\sqrt{2}$ accounts for the convention change from
$\kappa = 1/L_{\text{corr}}$ to $\kappa = \sqrt{2}/\ell$.

### Step 2: Match the prefactor

Using the operator factorisation from Step 1,

$$
\left(\kappa_{\text{new}}^2 \mathbf{I} - \mathbf{L}_h\right)^2
= \frac{1}{h^4}\left(\kappa_{\text{old}}^2 \mathbf{I} - \mathbf{L}\right)^2,
$$

so

$$
Q_{\text{new}} = \tau^2 h^2 \cdot \frac{1}{h^4}
  \left(\kappa_{\text{old}}^2 \mathbf{I} - \mathbf{L}\right)^2
= \frac{\tau^2}{h^2}
  \left(\kappa_{\text{old}}^2 \mathbf{I} - \mathbf{L}\right)^2.
$$

Setting this equal to $Q_{\text{old}}$:

$$
\frac{1}{\sigma_{\text{old}}^2} = \frac{\tau^2}{h^2}.
$$

Substituting $\tau = 1/(2\sqrt{\pi}\,\kappa_{\text{new}}\,\sigma_{\text{new}})$:

$$
\frac{1}{\sigma_{\text{old}}^2}
= \frac{1}{4\pi\,\kappa_{\text{new}}^2\,\sigma_{\text{new}}^2\, h^2}.
$$

Noting that $\kappa_{\text{new}} = \kappa_{\text{old}} / h = 1/(L_{\text{corr}} \cdot h)$:

$$
\frac{1}{\sigma_{\text{old}}^2}
= \frac{L_{\text{corr}}^2 h^2}{4\pi\,\sigma_{\text{new}}^2\, h^2}
= \frac{L_{\text{corr}}^2}{4\pi\,\sigma_{\text{new}}^2}.
$$

Solving for $\sigma_{\text{new}}$:

$$
\boxed{\sigma_{\text{new}} = \frac{\sigma_{\text{old}}\, L_{\text{corr}}}{2\sqrt{\pi}}.}
$$

Note that this result is **independent of the grid spacing** $h$.

---

## Summary

Given old parameters $L_{\text{corr}}$ (in grid cells) and $\sigma_{\text{old}}$, the
equivalent new parameters for grid spacing $h$ are:

| Old parameter | New parameter | Mapping |
|---|---|---|
| $L_{\text{corr}}$ (grid cells) | $\ell$ (physical units) | $\ell = \sqrt{2}\, L_{\text{corr}}\, h$ |
| $\sigma_{\text{old}}$ (scaling factor) | $\sigma_{\text{new}}$ (marginal std dev) | $\sigma_{\text{new}} = \sigma_{\text{old}}\, L_{\text{corr}} \,/\, (2\sqrt{\pi})$ |

The practical range of the Matérn field is $\rho = 2\ell = 2\sqrt{2}\, L_{\text{corr}}\, h$.

### Numerical example

Old parameters: $L_{\text{corr}} = 5$ grid cells, $\sigma_{\text{old}} = 0.02$ s/km, grid
spacing $h = 0.3°$:

$$
\ell = \sqrt{2} \times 5 \times 0.3 = 2.121°
$$

$$
\sigma_{\text{new}} = \frac{0.02 \times 5}{2\sqrt{\pi}} = \frac{0.1}{3.5449} = 0.02821 \text{ s/km}
$$

---

## References

Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian fields
and Gaussian Markov random fields: the stochastic partial differential equation approach.
*Journal of the Royal Statistical Society: Series B*, 73(4), 423–498.
