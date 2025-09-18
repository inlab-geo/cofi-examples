# Deterministic Inversion for the Neptune Problem

The discovery of Neptune is a celebrated example of using mathematical prediction to identify an unknown celestial body. Astronomers Urbain Le Verrier (France) and John Couch Adams (England) independently predicted the existence of Neptune based on irregularities observed in Uranus's orbit. Johann Galle later confirmed the prediction through observation.  

This framework illustrates how deterministic inversion can be applied to such problems: inferring the unknown parameters of a hidden celestial body from the observed perturbations it induces on a known planet's trajectory.  

---

## Problem Formulation

We aim to infer Neptune's parameters — its mass, velocity components, and position coordinates — by modeling Uranus's trajectory with and without Neptune's influence.  

The forward model is defined as:

$$
g(m) =
\begin{bmatrix}
\hat{x}_1(m) \\
\vdots \\
\hat{x}_N(m) \\
\hat{y}_1(m) \\
\vdots \\
\hat{y}_N(m) \\
\hat{z}_1(m) \\
\vdots \\
\hat{z}_N(m)
\end{bmatrix}
\in \mathbb{R}^{3N \times 1}
$$

where  
- $N$ is the number of data points,  
- $\hat{x}_j(m), \hat{y}_j(m), \hat{z}_j(m)$ are Uranus's predicted coordinates at observation $j$,  
- $m = (m_M, m_x, m_y, m_z, m_{v_x}, m_{v_y}, m_{v_z})$ are Neptune's parameters: mass, position coordinates, and velocity components.  

The observed data vector is:

$$
d =
\begin{bmatrix}
x_1 \\
\vdots \\
x_N \\
y_1 \\
\vdots \\
y_N \\
z_1 \\
\vdots \\
z_N
\end{bmatrix}
\in \mathbb{R}^{3N \times 1}
$$

The deterministic inversion problem becomes:

$$
\min_m \; \| g(m) - d \|_2^2
$$

---

## Physical Model: Newton's Law of Gravitation

The total gravitational force acting on planet $j$ due to other bodies is:

$$
\vec{F}_j = \sum_{i=1}^{9} G \frac{m_j m_i}{r_{ji}^2} \, \hat{r}_{ji}
$$

where  
- $m_j$: mass of the observed planet,  
- $m_i$: masses of the perturbing celestial bodies,  
- $r_{ji}$: distance between planets $j$ and $i$,  
- $\hat{r}_{ji}$: unit vector from planet $j$ to planet $i$,  
- $G$: gravitational constant.  

---

## Dynamical System

Let $\mathbf{r}$ be the stacked vector of all planetary positions and $\mathbf{a}$ their accelerations:

$$
\dot{\mathbf{r}} = \mathbf{v}, \quad \ddot{\mathbf{r}} = \mathbf{a}
$$

$$
\mathbf{r} =
\begin{bmatrix}
\mathbf{r}_1 \\
\vdots \\
\mathbf{r}_9
\end{bmatrix}, \quad
\mathbf{a} =
\begin{bmatrix}
\mathbf{a}_1 \\
\vdots \\
\mathbf{a}_9
\end{bmatrix}, \quad
\mathbf{r}, \mathbf{a} \in \mathbb{R}^{27 \times 1}
$$

This leads to the system of ODEs:

$$
\frac{d}{dt}
\begin{bmatrix}
\mathbf{r}(t) \\
\mathbf{v}(t)
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{v}(t) \\
\mathbf{a}(t)
\end{bmatrix}
$$

---

## Numerical Solution: Runge–Kutta 4 (RK4)

For the ODE

$$
\dot{f} = g(t, f), \quad f(t_0) = f_0,
$$

the RK4 update is:

$$
f_{n+1} = f_n + \frac{h}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right), \quad
t_{n+1} = t_n + h
$$

with

$$
\begin{aligned}
k_1 &= g(t_n, f_n), \\
k_2 &= g\left(t_n + \frac{h}{2}, f_n + \frac{h}{2}k_1\right), \\
k_3 &= g\left(t_n + \frac{h}{2}, f_n + \frac{h}{2}k_2\right), \\
k_4 &= g(t_n + h, f_n + h k_3).
\end{aligned}
$$

---

## Observational Noise

Observed coordinates are modeled with Gaussian noise:

$$
x_{\text{obs}} = x + \epsilon_x, \quad
y_{\text{obs}} = y + \epsilon_y, \quad
z_{\text{obs}} = z + \epsilon_z
$$

where

$$
\epsilon_x \sim \mathcal{N}(0, \sigma_x^2), \quad
\epsilon_y \sim \mathcal{N}(0, \sigma_y^2), \quad
\epsilon_z \sim \mathcal{N}(0, \sigma_z^2)
$$

Typical variances:

$$
\sigma_x = \sigma_y = 10^{-3}, \quad \sigma_z = 10^{-5}
$$

---

## Regularisation and Inversion

To prevent overfitting and ensure stable parameter estimation, regularisation is applied. The **L-curve method** helps select a balance between data misfit and model smoothness.  

### Levenberg–Marquardt Method

The Levenberg–Marquardt algorithm is widely used for nonlinear least-squares problems, such as:

$$
\min_m \; \| g(m) - d \|_2^2
$$

It combines two approaches:  
- **Gauss–Newton method** (efficient near the solution, but unstable when far),  
- **Gradient descent** (stable but slow).  

At each iteration, the update is:

$$
m_{k+1} = m_k - (J^T J + \lambda I)^{-1} J^T (g(m_k) - d),
$$

where  
- $J$ is the Jacobian of $g(m)$,  
- $\lambda$ is the damping parameter:  
  - large $\lambda$ → gradient descent behavior,  
  - small $\lambda$ → Gauss–Newton behavior.  

This adaptive balance allows the method to converge efficiently and robustly.  

---

## Summary

This theoretical framework combines:  
- Newtonian gravitational dynamics,  
- Numerical ODE solvers (RK4),  
- Observational error modeling,  
- Deterministic inversion with regularisation (Levenberg–Marquardt).  

Together, these tools enable the estimation of hidden planetary parameters — such as Neptune's mass, position, and velocity — from the observed perturbations in Uranus's orbit.