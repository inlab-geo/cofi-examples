# TypeError in birth perturbation with numpy >= 2.0 (Voronoi1D)

**bayesbay:** 0.3.8
**numpy:** 2.4.3
**Python:** 3.13

## Description

Any `Voronoi1D` trans-dimensional run crashes non-deterministically when a birth perturbation is proposed. This is a numpy 2.0 incompatibility.

`sample_site()` returns `np.random.uniform(vmin, vmax, spatial_dimensions)` which for the 1D case produces a shape-`(1,)` array. This is passed as `xp` to the Cython `nearest_neighbour_1d()`, which requires a scalar or 0-d array. numpy 2.0 removed the implicit 1-d → scalar coercion that Cython previously allowed.

## Traceback

```
File ".../bayesbay/discretization/_voronoi.py", line 529, in nearest_neighbour
    return nearest_neighbour_1d(
        xp=query_point, x=discretization, xlen=discretization.size
    )
File "src/bayesbay/_utils_1d.pyx", line 102, in bayesbay._utils_1d.nearest_neighbour_1d
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

## Root cause

`_voronoi.py` ~line 123:
```python
def sample_site(self) -> np.ndarray:
    return np.random.uniform(self.vmin, self.vmax, self.spatial_dimensions)
    # spatial_dimensions=1 → shape (1,), not a scalar
```

## Suggested fix

In `sample_site()`:
```python
result = np.random.uniform(self.vmin, self.vmax, self.spatial_dimensions)
return result.item() if self.spatial_dimensions == 1 else result
```

Or in `nearest_neighbour()`:
```python
return nearest_neighbour_1d(
    xp=float(query_point), x=discretization, xlen=discretization.size
)
```

## Minimal reproducer

```python
import numpy as np
import bayesbay
from bayesbay.discretization import Voronoi1D
from bayesbay.likelihood import Target, LogLikelihood
from bayesbay.parameterization import Parameterization
from bayesbay.prior import UniformPrior

print(f"bayesbay {bayesbay.__version__}, numpy {np.__version__}")

np.random.seed(0)
x_obs = np.linspace(0, 10, 20)
y_obs = np.sin(x_obs) + np.random.normal(0, 0.1, x_obs.size)

y_param = UniformPrior("y", vmin=-2, vmax=2, perturb_std=0.1)
pspace = Voronoi1D(
    name="v1d", vmin=0, vmax=10, perturb_std=0.5,
    n_dimensions=None, n_dimensions_min=1, n_dimensions_max=8,
    parameters=[y_param],
)
parameterization = Parameterization([pspace])

def forward(state):
    return np.interp(x_obs, state["v1d"]["discretization"], state["v1d"]["y"])

target = Target("d", y_obs, covariance_mat_inv=1.0 / 0.01)
log_likelihood = LogLikelihood(targets=target, fwd_functions=forward)

walkers_start = [parameterization.initialize() for _ in range(2)]
inversion = bayesbay.BaseBayesianInversion(
    walkers_starting_states=walkers_start,
    perturbation_funcs=parameterization.perturbation_funcs,
    log_like_ratio_func=log_likelihood,
    n_chains=2,
)
inversion.run(n_iterations=500, burnin_iterations=0, save_every=500, verbose=False)
```
