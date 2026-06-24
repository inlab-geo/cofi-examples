# Options for fixing bayesbay numpy 2.0 incompatibility

## Problem

bayesbay 0.3.8 crashes during Voronoi1D birth perturbations with numpy >= 2.0.
See `bayesbay_issue.md` for full details and a minimal reproducer.

## Options

### 1. Downgrade bayesbay to 0.3.7
Try installing an older release to see if the bug is present there.

```
pip install "bayesbay==0.3.7"
```

Then run the reproducer script to confirm:

```
python reproduce_bayesbay_birth_error.py
```

**Risk:** 0.3.7 was built against the same Cython code and may have the same
bug. All releases back to 0.1.x appear to share the same `_utils_1d.pyx`.

---

### 2. Pin numpy < 2.0 in the environment

```
pip install "numpy<2.0"
```

Restores the implicit 1-d → scalar coercion that Cython relied on.

**Risk:** Prevents using numpy 2.x features elsewhere; may conflict with other
packages that require numpy >= 2.0.

---

### 3. Monkey-patch bayesbay at runtime in the notebook

Add a cell before the bayesbay inversion cell in `notebooks/educator.py`:

```python
@app.cell
def _():
    from bayesbay.discretization._voronoi import Voronoi
    from bayesbay._utils_1d import nearest_neighbour_1d

    def _patched_nearest_neighbour(self, discretization, query_point):
        if self.spatial_dimensions == 1:
            return nearest_neighbour_1d(
                xp=float(query_point), x=discretization, xlen=discretization.size
            )
        import numpy as _np
        return _np.argmin(_np.linalg.norm(discretization - query_point, axis=1))

    Voronoi.nearest_neighbour = _patched_nearest_neighbour
    return
```

Does not modify any installed files. Only active for the duration of the
notebook process. Must be re-applied if bayesbay is updated.

---

### 4. Patch the installed bayesbay source directly

Edit `_voronoi.py` in the virtualenv:

```
~/.virtualenvs/cofi-reg/lib/python3.13/site-packages/bayesbay/discretization/_voronoi.py
```

Change `sample_site()` (~line 123):

```python
# Before:
return np.random.uniform(self.vmin, self.vmax, self.spatial_dimensions)

# After:
result = np.random.uniform(self.vmin, self.vmax, self.spatial_dimensions)
return result.item() if self.spatial_dimensions == 1 else result
```

**Risk:** Silently lost if bayesbay is reinstalled or upgraded.

---

### 5. Wait for an upstream fix

The issue has been filed on the bayesbay GitHub. If maintainers release a
patched version, a simple `pip install --upgrade bayesbay` resolves everything.

## Recommended approach

Try **Option 1** first (quickest to test). If 0.3.7 still crashes, apply
**Option 3** (monkey-patch) as a contained short-term fix while waiting for
**Option 5** (upstream release).
