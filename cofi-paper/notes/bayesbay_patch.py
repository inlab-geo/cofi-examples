"""Monkey-patches for bayesbay compatibility with NumPy 2.0+ and Python 3.14+.

Import this module before using bayesbay to apply all patches:

    import bayesbay_patch

Addresses two upstream issues:

- https://github.com/fmagrini/bayes-bay/issues/26
  NumPy 2.0+ removed implicit (1,)-array → scalar coercion.
  Voronoi1D.sample_site() returns shape (1,) for the 1D case, which the
  Cython nearest_neighbour_1d() rejects. Fixed by squeezing sample_site()
  output and converting xp via _to_scalar() before the Cython call.

- https://github.com/fmagrini/bayes-bay/issues/25
  Python 3.14 no longer implicitly converts numpy scalar types (np.intp,
  np.float64) to Python int/float when passed to Cython functions. Fixed
  by wrapping nearest_neighbour_1d, insert_1d, and delete_1d to explicitly
  convert arguments before forwarding to the Cython layer.

All patches are idempotent — safe to import multiple times.

Parallel-safe: a .pth file ensures this module is importable in joblib/loky
worker processes, and Sampler.advance_chain is wrapped to re-apply patches
in each worker before any chain iterations run. Works on both Linux
(forkserver) and macOS (spawn).
"""

import os
import site
import numpy as np
import bayesbay._utils_1d as _utils_1d
import bayesbay.discretization._voronoi as _voronoi_module
from bayesbay.discretization import Voronoi1D
from bayesbay.samplers._samplers import Sampler

_PATCHED = False


def _to_scalar(val):
    """Convert numpy scalar or 0-d/1-element array to Python scalar."""
    if isinstance(val, np.ndarray):
        return val.item()
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val


def _apply_patches():
    """Patch Cython functions and Voronoi1D.sample_site. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # --- Patch Cython functions in _utils_1d and _voronoi ---

    _orig_nn1d = _utils_1d.nearest_neighbour_1d

    def _patched_nn1d(xp, x, xlen):
        return _orig_nn1d(xp=_to_scalar(xp), x=x, xlen=int(xlen))

    _utils_1d.nearest_neighbour_1d = _patched_nn1d
    _voronoi_module.nearest_neighbour_1d = _patched_nn1d

    _orig_insert = _utils_1d.insert_1d

    def _patched_insert(values, index, value):
        return _orig_insert(values, int(index), _to_scalar(value))

    _utils_1d.insert_1d = _patched_insert
    _voronoi_module.insert_1d = _patched_insert

    _orig_delete = _utils_1d.delete_1d

    def _patched_delete(values, index):
        return _orig_delete(values, int(index))

    _utils_1d.delete_1d = _patched_delete
    _voronoi_module.delete_1d = _patched_delete

    # --- Patch Voronoi1D.sample_site for NumPy 2.0+ ---

    _orig_sample_site = Voronoi1D.sample_site

    def _patched_sample_site(self):
        result = _orig_sample_site(self)
        if self.spatial_dimensions == 1:
            return result.item()
        return result

    Voronoi1D.sample_site = _patched_sample_site


def _install_pth():
    """Install a .pth file so bayesbay_patch is importable in worker processes.

    Writes a one-line .pth file into site-packages that adds the notebook
    directory to sys.path. This is sufficient for workers to ``import
    bayesbay_patch`` when Sampler.advance_chain triggers it.

    The ``import bayesbay_patch`` line in the .pth intentionally fails
    at early startup (dependency ordering) — it is harmless. The actual
    patching in workers is done by the wrapped Sampler.advance_chain.
    """
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
    pth_name = "bayesbay_patch.pth"

    site_packages = None
    for p in site.getsitepackages():
        if os.path.isdir(p):
            site_packages = p
            break
    if site_packages is None:
        return

    pth_path = os.path.join(site_packages, pth_name)
    content = f"{notebook_dir}\n"

    try:
        if os.path.exists(pth_path):
            with open(pth_path) as f:
                if f.read() == content:
                    return
        with open(pth_path, "w") as f:
            f.write(content)
    except OSError:
        pass


def _patch_sampler():
    """Wrap Sampler.advance_chain so workers re-apply patches.

    joblib/loky workers are fresh Python processes that re-import all
    modules without our patches. We wrap Sampler.advance_chain (which
    runs in the MAIN process and dispatches to workers) to replace the
    function passed to joblib.delayed with a wrapper that imports
    bayesbay_patch in the worker before executing the chain.
    """
    _orig_advance_chain = Sampler.advance_chain

    def _patched_advance_chain(self, n_iterations, burnin_iterations=0,
                               save_every=100, verbose=True, print_every=100):
        from functools import partial
        import joblib
        from bayesbay._markov_chain import MarkovChain

        func = partial(
            MarkovChain.advance_chain,
            n_iterations=n_iterations,
            burnin_iterations=burnin_iterations,
            save_every=save_every,
            verbose=verbose,
            print_every=print_every,
            begin_iteration=self.begin_iteration,
            end_iteration=self.end_iteration,
        )

        def _worker_func(chain, _f=func):
            import bayesbay_patch  # noqa: F811 — re-apply patches in worker
            return _f(chain)

        if self.parallel_config.get("n_jobs", 1) > 1:
            self._chains = joblib.Parallel(**self.parallel_config)(
                joblib.delayed(_worker_func)(chain) for chain in self.chains
            )
        else:
            self._chains = [func(chain) for chain in self.chains]
        self.on_end_advance_chain()
        return self.chains

    Sampler.advance_chain = _patched_advance_chain


_apply_patches()
_install_pth()
_patch_sampler()
