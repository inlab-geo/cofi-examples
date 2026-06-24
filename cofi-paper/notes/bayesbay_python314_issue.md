# BayesBay fails on Python 3.14: Cython functions reject numpy scalar types

## Summary

BayesBay 0.3.7 is incompatible with Python 3.14 due to stricter type coercion rules. Cython functions in `_utils_1d` no longer accept numpy scalar types (`np.intp`, `np.float64`) which were implicitly converted in Python 3.13 and earlier.

## Environment

- **Python**: 3.14.2
- **BayesBay**: 0.3.7
- **NumPy**: 2.4.2
- **OS**: Fedora 43 (Linux 6.18.7)

## Minimal Reproduction

```python
import numpy as np
from bayesbay._utils_1d import nearest_neighbour_1d

x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
query_point = np.float64(4.0)  # numpy scalar
xlen = np.intp(len(x))  # numpy scalar

# Fails on Python 3.14
result = nearest_neighbour_1d(xp=query_point, x=x, xlen=xlen)
```

See attached `reproduce_python314_bug.py` for complete reproduction script.

## Error Messages

```
TypeError: an integer is required
```

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

## Affected Functions

| Function | Parameter | Expected | Receives |
|----------|-----------|----------|----------|
| `nearest_neighbour_1d` | `xp` | `float` | `np.float64` |
| `nearest_neighbour_1d` | `xlen` | `int` | `np.intp` |
| `insert_1d` | `index` | `int` | `np.intp` |
| `insert_1d` | `value` | `float` | `np.float64` |
| `delete_1d` | `index` | `int` | `np.intp` |

## Root Cause

Python 3.14 has stricter type coercion rules. The Cython-compiled functions in `bayesbay/_utils_1d.pyx` expect Python `int` and `float` types, but receive numpy scalar types from array operations.

In Python 3.13 and earlier, numpy scalars were implicitly converted to Python scalars when passed to Cython functions. Python 3.14 requires explicit conversion.

## Suggested Fix

Option 1: Update Cython function signatures to handle numpy types:

```cython
# In _utils_1d.pyx
def nearest_neighbour_1d(xp, double[::1] x, xlen):
    cdef double _xp = float(xp)
    cdef int _xlen = int(xlen)
    # ... rest of function
```

Option 2: Add type conversion in Python wrapper functions before calling Cython code.

## Workaround

Users can apply this monkey-patch before using BayesBay:

```python
import numpy as np
import bayesbay._utils_1d as _utils_1d
import bayesbay.discretization._voronoi as _voronoi_module
from bayesbay.discretization import Voronoi1D

def _to_scalar(val):
    """Convert numpy scalar to Python scalar."""
    if isinstance(val, np.ndarray):
        return val.item()
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    return val

# Patch nearest_neighbour_1d
_orig_nn1d = _utils_1d.nearest_neighbour_1d
def _patched_nn1d(xp, x, xlen):
    return _orig_nn1d(xp=_to_scalar(xp), x=x, xlen=int(xlen))
_utils_1d.nearest_neighbour_1d = _patched_nn1d
_voronoi_module.nearest_neighbour_1d = _patched_nn1d

# Patch insert_1d
_orig_insert = _utils_1d.insert_1d
def _patched_insert(values, index, value):
    return _orig_insert(values, int(index), _to_scalar(value))
_utils_1d.insert_1d = _patched_insert
_voronoi_module.insert_1d = _patched_insert

# Patch delete_1d
_orig_delete = _utils_1d.delete_1d
def _patched_delete(values, index):
    return _orig_delete(values, int(index))
_utils_1d.delete_1d = _patched_delete
_voronoi_module.delete_1d = _patched_delete
```

**Note:** When using parallel processing (`n_jobs > 1`), worker processes don't inherit the monkey-patch. Use `parallel_config={"n_jobs": 1}` to run serially.

## Additional Notes

There's also a harmless semaphore leak warning on Python 3.14 shutdown related to loky/joblib:

```
UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

This is cosmetic and doesn't affect results.
