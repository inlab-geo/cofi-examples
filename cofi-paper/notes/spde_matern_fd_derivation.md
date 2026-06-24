# Deriving the Matérn SPDE Precision Matrix via Finite Differences

## Overview

This note provides a self-contained derivation of the precision matrix $Q$ and its
sparse factor $\mathbf{R}$ for a Matérn $\nu=1$ Gaussian random field on a regular 2D
grid, using **only finite-difference** concepts. The result matches the finite-element
derivation of Lindgren, Rue & Lindström (2011) and is the basis for the
`cofi.utils.SPDEMaternReg` implementation.

---

## 1. Discretising Spatial White Noise

Let $\mathcal{W}(\mathbf{s})$ be continuous Gaussian white noise in 2D, with zero mean
and Dirac delta covariance:

$$
\operatorname{cov}\bigl(\mathcal{W}(\mathbf{s}),\, \mathcal{W}(\mathbf{s}')\bigr) = \delta(\mathbf{s} - \mathbf{s}').
$$

Continuous white noise cannot be evaluated pointwise — it must be integrated over a
spatial volume. On a regular grid with spacing $(h_x, h_y)$, each node $i$ owns a cell
of area $V_i = h_x h_y$. We define the discrete white noise at node $i$ as the
**cell average**:

$$
w_i = \frac{1}{V_i} \int_{V_i} \mathcal{W}(\mathbf{s})\, d\mathbf{s}.
$$

Since the integrals of white noise over disjoint cells are independent, the $w_i$ are
independent Gaussian variables. Their variance is:

$$
\operatorname{var}(w_i)
= \frac{1}{V_i^2} \int_{V_i} \int_{V_i} \underbrace{\mathbb{E}\bigl[\mathcal{W}(\mathbf{s})\,\mathcal{W}(\mathbf{s}')\bigr]}_{\delta(\mathbf{s}-\mathbf{s}')}\, d\mathbf{s}\, d\mathbf{s}'
= \frac{1}{V_i^2} \int_{V_i} 1\, d\mathbf{s}
= \frac{V_i}{V_i^2}
= \frac{1}{h_x h_y}.
$$

In vector form:

$$
\mathbf{w} \sim \mathcal{N}\!\left(\mathbf{0},\; \frac{1}{h_x h_y}\,\mathbf{I}\right).
$$

---

## 2. Discretising the SPDE

A Matérn random field $X(\mathbf{s})$ with smoothness $\nu$ satisfies the SPDE
(Whittle 1954):

$$
(\kappa^2 - \Delta)^{\alpha/2}\bigl[\tau\, X(\mathbf{s})\bigr] = \mathcal{W}(\mathbf{s}),
$$

where $\alpha = \nu + d/2$, $\kappa = \sqrt{2}/\ell$ is the inverse length scale, and
$\tau$ controls the marginal variance. For $\nu = 1$ in $d = 2$ dimensions,
$\alpha = 2$ and $\alpha/2 = 1$, so the SPDE reduces to

$$
\tau\,(\kappa^2 - \Delta)\, X(\mathbf{s}) = \mathcal{W}(\mathbf{s}).
$$

Replace the continuous Laplacian $\Delta$ with the discrete finite-difference Laplacian
$\mathbf{L}_h$ (which already incorporates the $1/h^2$ scaling in each direction). Let
$\mathbf{X}$ be the vector of field values at grid nodes. Defining the discrete SPDE
operator

$$
\mathbf{A} = \kappa^2 \mathbf{I} - \mathbf{L}_h,
$$

the discrete system becomes

$$
\tau\, \mathbf{A}\, \mathbf{X} = \mathbf{w}.
$$

Since $\mathbf{L}_h$ is symmetric, $\mathbf{A}$ is also symmetric.

---

## 3. Deriving the Covariance and Precision Matrices

Isolating $\mathbf{X}$:

$$
\mathbf{X} = \frac{1}{\tau}\, \mathbf{A}^{-1}\, \mathbf{w}.
$$

Applying the covariance transformation $\operatorname{cov}(\mathbf{M}\mathbf{v}) = \mathbf{M}\operatorname{cov}(\mathbf{v})\mathbf{M}^\top$:

$$
\boldsymbol{\Sigma} = \operatorname{cov}(\mathbf{X})
= \frac{1}{\tau^2}\, \mathbf{A}^{-1}\, \operatorname{cov}(\mathbf{w})\, \mathbf{A}^{-\top}.
$$

Substituting $\operatorname{cov}(\mathbf{w}) = \frac{1}{h_x h_y}\,\mathbf{I}$ and using
$\mathbf{A}^{-\top} = \mathbf{A}^{-1}$ (symmetry):

$$
\boldsymbol{\Sigma} = \frac{1}{\tau^2\, h_x h_y}\, \mathbf{A}^{-2}.
$$

The precision matrix is the inverse of the covariance:

$$
\boxed{
Q = \boldsymbol{\Sigma}^{-1} = \tau^2\, h_x h_y\, \mathbf{A}^2
  = \tau^2\, h_x h_y\, (\kappa^2 \mathbf{I} - \mathbf{L}_h)^2.
}
$$

---

## 4. Extracting the Sparse Factor $\mathbf{R}$

We seek $\mathbf{R}$ such that $Q = \mathbf{R}^\top \mathbf{R}$. Since $\mathbf{A}$ is
symmetric:

$$
Q = \bigl(\tau\sqrt{h_x h_y}\;\mathbf{A}\bigr)^\top
    \bigl(\tau\sqrt{h_x h_y}\;\mathbf{A}\bigr),
$$

so

$$
\boxed{
\mathbf{R} = \tau\sqrt{h_x h_y}\;(\kappa^2 \mathbf{I} - \mathbf{L}_h).
}
$$

This is a **sparse** matrix: $\mathbf{L}_h$ has at most 5 non-zeros per row (the node
and its four grid neighbours), so $\mathbf{R}$ has the same sparsity pattern. The
precision matrix $Q = \mathbf{R}^\top \mathbf{R}$ has at most 13 non-zeros per row —
vastly more efficient than the dense covariance $\boldsymbol{\Sigma}$.

---

## 5. Connection to the FEM Derivation

In the Galerkin finite-element formulation of Lindgren et al. (2011), the precision
matrix takes the form

$$
Q_{\text{FEM}} = \tau^2\, \mathbf{K}^\top \mathbf{M}^{-1} \mathbf{K},
$$

where $\mathbf{M}$ is the mass matrix and $\mathbf{K} = \kappa^2 \mathbf{M} + \mathbf{G}$
(with $\mathbf{G}$ the stiffness matrix).

On a regular grid, the finite-difference scheme is equivalent to using a **lumped mass
matrix** $\mathbf{M} = h_x h_y\, \mathbf{I}$. Under this approximation:

- The stiffness matrix satisfies $\mathbf{G} = -\mathbf{M}\,\mathbf{L}_h = -h_x h_y\, \mathbf{L}_h$.
- The SPDE operator becomes $\mathbf{K} = \kappa^2 h_x h_y\, \mathbf{I} - h_x h_y\, \mathbf{L}_h = h_x h_y\, \mathbf{A}$.
- Substituting into $Q_{\text{FEM}}$:

$$
Q_{\text{FEM}} = \tau^2\, (h_x h_y)^2\, \mathbf{A}^\top \cdot \frac{1}{h_x h_y}\, \mathbf{I} \cdot \mathbf{A}
= \tau^2\, h_x h_y\, \mathbf{A}^2 = Q.
$$

The FD and FEM derivations agree exactly on a uniform grid with lumped mass. The area
factor $h_x h_y$ in the FD result arises from the variance of cell-averaged white noise
and plays exactly the same role as $\mathbf{M}^{-1}$ in the FEM formulation.

---

## 6. SPDE Parameters and Physical Interpretation

| Parameter | Definition | Interpretation |
|---|---|---|
| $\ell$ | Matérn length scale (physical units) | Distance scale of spatial correlation |
| $\kappa = \sqrt{2}/\ell$ | Inverse length scale | SPDE wavenumber |
| $\rho = 2\ell$ | Practical range | Distance where correlation $\approx 0.14$ |
| $\sigma$ | Marginal standard deviation | Amplitude of field fluctuations |
| $\tau = \frac{1}{2\sqrt{\pi}\,\kappa\,\sigma}$ | Precision amplitude | Controls $Q$ magnitude |

The marginal variance in the continuous limit is

$$
\sigma^2 = \frac{1}{4\pi\,\kappa^2\,\tau^2},
$$

which follows from the spectral density of the Matérn field with $\nu=1$ in 2D.

---

## References

- Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian
  fields and Gaussian Markov random fields: the stochastic partial differential equation
  approach. *Journal of the Royal Statistical Society: Series B*, 73(4), 423–498.
- Whittle, P. (1954). On stationary processes in the plane. *Biometrika*, 41(3/4), 434–449.
