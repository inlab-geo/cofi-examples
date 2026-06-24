"""
Minimal reproducer for bayesbay birth-perturbation crash with numpy >= 2.0.

bayesbay version: 0.3.8
numpy version:    2.4.3

The crash occurs during a birth perturbation in Voronoi1D trans-dimensional
sampling. sample_site() returns a shape-(1,) array for the 1D case, which is
then passed as `xp` to the Cython function nearest_neighbour_1d(). numpy 2.0
removed the implicit 1-d → scalar coercion that Cython relied on, causing:

    TypeError: only 0-dimensional arrays can be converted to Python scalars

Root cause in bayesbay/discretization/_voronoi.py (~line 123):

    def sample_site(self) -> np.ndarray:
        return np.random.uniform(self.vmin, self.vmax, self.spatial_dimensions)

For spatial_dimensions=1 this produces shape (1,), not a scalar. The result
flows into nearest_neighbour() (~line 529) and then into the Cython
nearest_neighbour_1d(), which cannot accept a 1-d array as xp.

A minimal fix in _voronoi.py would be to squeeze the result for the 1D case:

    def sample_site(self):
        result = np.random.uniform(self.vmin, self.vmax, self.spatial_dimensions)
        return result.item() if self.spatial_dimensions == 1 else result

Or equivalently in nearest_neighbour():

    def nearest_neighbour(self, discretization, query_point):
        if self.spatial_dimensions == 1:
            return nearest_neighbour_1d(
                xp=float(query_point), x=discretization, xlen=discretization.size
            )
        ...
"""

import numpy as np
import bayesbay
from bayesbay.discretization import Voronoi1D
from bayesbay.likelihood import Target, LogLikelihood
from bayesbay.parameterization import ParameterSpace, Parameterization
from bayesbay.prior import UniformPrior
print(f"bayesbay {bayesbay.__version__}, numpy {np.__version__}")
print()

# --- Minimal 1D trans-dimensional problem ---
# Fit a piecewise-linear curve to a handful of synthetic data points.

np.random.seed(0)
x_obs = np.linspace(0, 10, 20)
y_obs = np.sin(x_obs) + np.random.normal(0, 0.1, x_obs.size)
sigma = 0.1

y_param = UniformPrior("y", vmin=-2, vmax=2, perturb_std=0.1)

pspace = Voronoi1D(
    name="v1d",
    vmin=0,
    vmax=10,
    perturb_std=0.5,
    n_dimensions=None,
    n_dimensions_min=1,
    n_dimensions_max=8,
    parameters=[y_param],
)

parameterization = Parameterization([pspace])


def forward(state):
    sites = state["v1d"]["discretization"]
    vals  = state["v1d"]["y"]
    return np.interp(x_obs, sites, vals)


target = Target("d", y_obs, covariance_mat_inv=1.0 / sigma**2)
log_likelihood = LogLikelihood(targets=target, fwd_functions=forward)

walkers_start = [parameterization.initialize() for _ in range(2)]

inversion = bayesbay.BaseBayesianInversion(
    walkers_starting_states=walkers_start,
    perturbation_funcs=parameterization.perturbation_funcs,
    log_like_ratio_func=log_likelihood,
    n_chains=2,
)

print("Running sampler — will crash during a birth perturbation ...")
# Birth proposals happen randomly; a few hundred iterations is enough to trigger one.
inversion.run(n_iterations=500, burnin_iterations=0, save_every=500, verbose=False)
print("Done (no crash).")
