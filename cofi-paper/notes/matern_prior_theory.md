# Gaussian Prior Covariance and the Matérn Alternative for 2D Spatial Fields

## Overview

In Bayesian geophysical inversion, prior knowledge about the model is encoded in a prior
probability distribution. For continuous spatial fields — such as seismic velocity or
slowness distributed over a 2D study region — the prior is commonly taken to be Gaussian,
characterised by a mean (reference model) and a covariance function that encodes the expected
degree of spatial correlation. This note reviews the Gaussian prior formulation, discusses
the computational bottleneck that arises from the non-sparse nature of the resulting covariance
matrix, and introduces the Matérn covariance as an attractive alternative that retains computational efficiency through a sparse precision matrix.

---

## The Gaussian Prior and its Covariance Matrix

Consider a model vector $\mathbf{m} \in \mathbb{R}^n$ representing $n$ spatially distributed
parameters (e.g. slowness at each node of a 2D grid). A Gaussian prior with reference model
$\mathbf{m}_0$ is written

$$
p(\mathbf{m}) \propto \exp\!\left(-\tfrac{1}{2}(\mathbf{m} - \mathbf{m}_0)^\top C_m^{-1}(\mathbf{m} - \mathbf{m}_0)\right),
$$

where $C_m \in \mathbb{R}^{n \times n}$ is the prior model covariance matrix. For a stationary
field, the $(i,j)$ entry of $C_m$ depends only on the distance $r_{ij}$ between nodes $i$ and $j$:

$$
[C_m]_{ij} = \sigma^2\, \rho(r_{ij}),
$$

with $\sigma^2$ the marginal variance and $\rho(r)$ a positive-definite correlation function
satisfying $\rho(0) = 1$. Common choices include the Gaussian $\rho(r) = \exp(-r^2 / 2L^2)$
and the exponential $\rho(r) = \exp(-r/L)$, where $L$ is a correlation length.

Incorporating this prior into the objective function alongside the data misfit gives

$$
\Phi(\mathbf{m}) = \bigl(\mathbf{d} - \mathbf{f}(\mathbf{m})\bigr)^\top C_d^{-1} \bigl(\mathbf{d} - \mathbf{f}(\mathbf{m})\bigr)
+ (\mathbf{m} - \mathbf{m}_0)^\top C_m^{-1} (\mathbf{m} - \mathbf{m}_0),
$$

where $\mathbf{d}$ is the data vector, $\mathbf{f}(\mathbf{m})$ the forward operator, and
$C_d$ the data covariance. The gradient and Hessian of $\Phi$ are required by gradient-based
optimisers. With Jacobian $\mathbf{J} = \partial \mathbf{f}/\partial \mathbf{m}$, these are

$$
\nabla \Phi = -\mathbf{J}^\top C_d^{-1}\bigl(\mathbf{d} - \mathbf{f}(\mathbf{m})\bigr) + C_m^{-1}(\mathbf{m} - \mathbf{m}_0),
$$

$$
\mathbf{H} \approx \mathbf{J}^\top C_d^{-1} \mathbf{J} + C_m^{-1},
$$

where the second line uses the Gauss-Newton approximation (dropping second-order terms in
the residual).

---

## Computational Limitations of Dense Prior Covariance

The fundamental difficulty with the Gaussian prior as formulated above is that $C_m$ is
**dense**: every pair of nodes has a non-zero (if small) prior correlation, so $C_m$ requires
$O(n^2)$ storage. For a 2D grid with $n = 150 \times 130 \approx 19{,}500$ nodes, $C_m$ has
roughly $3.8 \times 10^8$ entries, occupying approximately 3 GB in double precision.

More critically, the regularisation term in the gradient and Hessian requires $C_m^{-1}$,
not $C_m$. Even if $C_m$ is constructed with its elements, 
$[C_m]_{ij}$, set to zero for cells at large separations, $r>r_{crit}$, which creates a sparse $C_m$, its inverse  $C_m^{-1}$ remains **dense** and costs $O(n^3)$ to compute by direct methods. For the grid
sizes typical of regional seismic tomography this is prohibitive.

A further consequence is that $C_m^{-1}$ cannot be stored or applied sparsely, so the
regularisation term $C_m^{-1}(\mathbf{m} - \mathbf{m}_0)$ densifies the gradient and Hessian
even when the data Jacobian $\mathbf{J}$ is sparse.

---

## The Matérn Covariance Function

The Matérn family of covariance functions provides a flexible and theoretically well-grounded
alternative. The Matérn covariance with smoothness parameter $\nu$ and correlation length
$L_\text{corr}$ is

$$
C(r) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\, r}{L_\text{corr}}\right)^\nu K_\nu\!\left(\frac{\sqrt{2\nu}\, r}{L_\text{corr}}\right),
$$

where $K_\nu$ is the modified Bessel function of the second kind of order $\nu$. The parameter
$\nu$ controls the differentiability of the field: $\nu = 1/2$ yields the exponential
covariance (continuous but not differentiable), $\nu = 3/2$ gives once-differentiable fields,
and $\nu \to \infty$ recovers the Gaussian covariance (infinitely differentiable). The case
$\nu = 1$ is a useful intermediate: once-differentiable fields with the closed-form correlation

$$
\rho(r) = \frac{r}{L_\text{corr}}\, K_1\!\left(\frac{r}{L_\text{corr}}\right).
$$

This function decays from $\rho(0) = 1$ and reaches $\rho(L_\text{corr}) \approx 0.60$,
so $L_\text{corr}$ is the distance at which the prior correlation falls to roughly 60%.
Beyond $L_\text{corr}$, correlation decays rapidly:
$\rho(2L_\text{corr}) \approx 0.28$, $\rho(3L_\text{corr}) \approx 0.12$.

Figure 1 compares the Matérn ν=1 correlation function with the squared exponential
(Gaussian) and the exponential (Matérn ν=½). All three are plotted against the normalised
distance $r/L_\text{corr}$, parameterised so that the correlation at $r = L_\text{corr}$
is comparable. The log-scale panel (right) reveals the key qualitative difference in tail
behaviour: the squared exponential decays super-exponentially (fastest), Matérn ν=1 decays
algebraically-exponentially (intermediate), and the exponential decays slowest. A practical
consequence is that the squared exponential prior produces very smooth fields but assigns
negligible prior probability to any long-range structure, while the Matérn ν=1 retains
moderate long-range correlation and generates fields that are once but not twice
mean-square differentiable — a physically reasonable assumption for regional geophysical
parameters. Note also that the exponential correlation function (Matérn ν=½) is the one
actually implemented by the `GaussianPrior` utility (via `exp(-r/L)`, the dense
covariance construction described in the previous section), not the squared exponential.

<figure>
<img src="../figures/correlation_comparison.png" alt="Comparison of Gaussian, Matérn ν=1 and exponential correlation functions on linear and log scales">
<figcaption>Figure 1. Correlation functions as a function of normalised distance r/L<sub>corr</sub>. Left: linear scale. Right: log scale, showing the qualitatively different tail behaviour. Markers indicate the correlation value at r = L<sub>corr</sub> for each function. The squared exponential (Gaussian) and Matérn ν=1 both reach ρ ≈ 0.60 at r = L<sub>corr</sub>, making L<sub>corr</sub> directly comparable between the two. The exponential (Matérn ν=½) reaches only ρ ≈ 0.37 at the same distance.</figcaption>
</figure>

---

## Sparse Precision via the SPDE Representation

The key insight that enables efficient computation is due to Lindgren, Rue & Lindström (2011),
who showed that a Matérn Gaussian random field $u(\mathbf{x})$ on $\mathbb{R}^d$ can be
represented as the solution to the stochastic partial differential equation (SPDE)

$$
(\kappa^2 - \Delta)^{\alpha/2}\, u(\mathbf{x}) = \mathcal{W}(\mathbf{x}),
$$

where $\Delta$ is the Laplacian, $\mathcal{W}$ is Gaussian white noise, $\kappa = 1/L_\text{corr}$,
and $\alpha = \nu + d/2$. For $\nu = 1$ in two dimensions ($d = 2$), we have $\alpha = 2$,
so the operator factorises as

$$
(\kappa^2 - \Delta)^2\, u = \mathcal{W}.
$$

The precision matrix (inverse covariance) of the discretised field is

$$
Q = \tau^2 (\kappa^2 \mathbf{I} - \mathbf{L})^2,
$$

where $\mathbf{L}$ is the discrete 2D Laplacian and $\tau$ is a precision scaling parameter
related to $\sigma$ via $\sigma^2 \approx 1 / (4\pi \kappa^2 \tau^2)$ in the continuous limit.
Crucially, $Q$ is the **square of a sparse matrix** and is itself sparse: $\mathbf{L}$
has at most five non-zero entries per row (the node itself and its four grid neighbours),
so $(\kappa^2 \mathbf{I} - \mathbf{L})^2$ has at most 13 non-zero entries per row. For
$n = 19{,}500$ nodes this represents a reduction from $\sim 3.8 \times 10^8$ entries (dense)
to $\sim 2.5 \times 10^5$ entries (sparse) — a factor of roughly 1500.

---

## Incorporating the Sparse Prior into the Objective

In practice it is not necessary to form $Q$ explicitly. On a uniform unit grid, one can
write $Q = \mathbf{R}^\top \mathbf{R}$ with the sparse factor

$$
\mathbf{R} = \tau (\kappa^2 \mathbf{I} - \mathbf{L}).
$$

For a rectangular grid with spacings $(h_x, h_y)$, the lumped-mass discretisation used
in `cofi.utils.SPDEMaternReg` is

$$
Q = \tau^2 \mathbf{B}_h^\top \mathbf{M} \mathbf{B}_h,
\qquad
\mathbf{B}_h = \kappa^2 \mathbf{I} - \mathbf{L}_h,
\qquad
\mathbf{M} \approx h_x h_y \mathbf{I},
$$

so the sparse factor becomes

$$
\mathbf{R} = \tau \mathbf{M}^{1/2} \mathbf{B}_h
= \tau \sqrt{h_x h_y}\,(\kappa^2 \mathbf{I} - \mathbf{L}_h).
$$

This reduces to the unit-grid expression above when $h_x = h_y = 1$. In the CoFI API,
the public parameter is `rho`, the Matérn practical range, with $\kappa = \sqrt{8}/\rho$;
the older notation $L_\text{corr} = 1/\kappa$ is only the unit-correlation-length shorthand.

the prior regularisation term becomes

$$
(\mathbf{m} - \mathbf{m}_0)^\top Q\, (\mathbf{m} - \mathbf{m}_0) = \|\mathbf{R}(\mathbf{m} - \mathbf{m}_0)\|^2.
$$

This allows the full regularised objective to be written as a least-squares problem in
augmented form:

$$
\Phi(\mathbf{m}) = \left\|\begin{bmatrix} \mathbf{d} - \mathbf{f}(\mathbf{m}) \\ \sqrt{\mu}\,\mathbf{R}(\mathbf{m} - \mathbf{m}_0) \end{bmatrix}\right\|^2,
$$

where $\mu$ is a scalar regularisation weight that balances data fit against prior
regularisation and is typically selected by L-curve analysis. The augmented Jacobian is

$$
\mathbf{J}_\text{aug} = \begin{bmatrix} \mathbf{J} \\ \sqrt{\mu}\,\mathbf{R} \end{bmatrix},
$$

and because both $\mathbf{J}$ (for ray-based forward operators, each ray intersects only a
small fraction of grid cells) and $\mathbf{R}$ are sparse, $\mathbf{J}_\text{aug}$ can be
stored and manipulated in sparse format throughout.

The gradient and Gauss-Newton Hessian of $\Phi$ are

$$
\nabla \Phi = -\mathbf{J}^\top(\mathbf{d} - \mathbf{f}(\mathbf{m})) + \mu\,\mathbf{R}^\top \mathbf{R}(\mathbf{m} - \mathbf{m}_0),
$$

$$
\mathbf{H} \approx \mathbf{J}^\top \mathbf{J} + \mu\,\mathbf{R}^\top \mathbf{R}.
$$

Both $\mathbf{R}^\top \mathbf{R}$ and $\mathbf{J}^\top \mathbf{J}$ are sparse, so the
Hessian can be assembled and factored without ever forming a dense $n \times n$ matrix.
This contrasts with the dense Gaussian prior, where $C_m^{-1}$ densifies the Hessian
regardless of the sparsity of $\mathbf{J}$.

---

## Summary

| Property | Dense Gaussian prior | Matérn SPDE prior |
|---|---|---|
| Covariance $C_m$ | Dense, $O(n^2)$ storage | Not formed explicitly |
| Precision $C_m^{-1}$ | Dense, $O(n^3)$ to compute | Sparse, $O(n)$ storage |
| Gradient term | Dense matrix-vector product | Sparse matrix-vector product |
| Hessian | Densified by $C_m^{-1}$ | Remains sparse |
| Free parameters | $\sigma$, $L_\text{corr}$ | $\sigma$, $L_\text{corr}$, $\mu$ |
| Smoothness | Controlled by choice of $\rho$ | Controlled by $\nu$ (fixed at 1 here) |

The Matérn SPDE approach provides the same two physically interpretable parameters
($L_\text{corr}$ and $\sigma$) as an explicitly constructed dense covariance, while
maintaining sparsity in the precision matrix and enabling scalable computation for large
2D grids.

---

## References

Lindgren, F., Rue, H., & Lindström, J. (2011). An explicit link between Gaussian fields
and Gaussian Markov random fields: the stochastic partial differential equation approach.
*Journal of the Royal Statistical Society: Series B*, 73(4), 423–498.
https://doi.org/10.1111/j.1467-9868.2011.00777.x
