# Electrical Resistivity Tomography with PyGIMLi and CoFI

Welcome to the ERT example!

This folder contains a Jupyter notebook file, some self-contained example scripts and
sample results.

- [pygimli_ert.ipynb](pygimli_ert.ipynb) is the entry point and contains some explanation around
  the problem.
- [pygimli_ert_lib.py](pygimli_ert_lib.py) is the library file that's imported by all the other 
  scripts and the notebook. You don't have to touch this file unless you are curious about the
  details of defining the ERT problem with PyGIMLi, or you'd like to customise your own problem.
- other files are mainly example scripts - each script contains one single ERT problem (either
  rectangular mesh or triangular mesh) solved with one selected inversion solver.
  - [pygimli_ert_tri_inbuilt_invert.py](pygimli_ert_tri_inbuilt_invert.py) runs inversion with a
    triangular mesh by PyGIMLi's inbuilt solver. This example doesn't use an inversion approached
    binded by CoFI, and is for reference only.
  - [pygimli_ert_tri_scipy_min.py](pygimli_ert_tri_scipy_min.py) runs inversion with a triangular
    mesh by SciPy's optimizers.
  - [pygimli_ert_tri_gauss_newton.py](pygimli_ert_tri_gauss_newton.py) runs inversion with a 
    triangular mesh by a Gauss Newton's optimization approach handwritten by us.
  - [pygimli_ert_tri_gauss_newton_armijo_linesearch.py](pygimli_ert_tri_gauss_newton_armijo_linesearch.py)
    is an update to the above Gauss Newton example, with the addition of a line search functionality
    to increase the speed of convergence.
  - Files with `rect` instead of `tri` in their names follow the same idea as the four files above,
    with the difference that they use a rectangular mesh for inversion, instead of a triangular one.
- `figs` folder contains all the sample results from running the scripts listed above.

### pyGIMLi notebooks (Python compatibility)

Some notebooks depend on **pyGIMLi (pygimli)**. At the time of writing, pyGIMLi’s compiled core (`pgcore`) does not provide wheels for **Python 3.13**, so these notebooks will not run on 3.13 unless you build from source.

**Recommended:** use Python **3.12** (or 3.11) for pyGIMLi-based notebooks.  
**Alternative:** build pyGIMLi/pgcore from source (advanced).
