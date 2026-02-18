# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoFI Examples is the example/tutorial repository for [CoFI (Common Framework for Inference)](https://github.com/inlab-geo/cofi), an open-source framework for interfacing between generic inference algorithms and geoscience problems. The repo contains Jupyter notebooks and Python scripts demonstrating CoFI applied to problems ranging from simple linear regression to complex geoscience inversions.

Related projects: [CoFI](https://github.com/inlab-geo/cofi), [Espresso](https://github.com/inlab-geo/espresso)

## Environment Setup

```bash
mamba env create -f envs/environment.yml
mamba activate cofi_env
```

Key dependencies: `cofi` (inference framework), `geo-espresso` (geophysics test problems), `pygimli`, `seislib`, `scipy`, `matplotlib`.

## Common Commands

All commands assume project root as working directory.

**Run notebooks interactively:**
```bash
jupyter-lab
```

**Create a new example scaffold:**
```bash
python tools/generate_example/create_new_example.py <example-name>
```

**Run all notebooks (CI validation):**
```bash
python tools/run_notebooks/run_notebooks.py all
```

**Run all example scripts (integration/regression tests):**
```bash
python tools/validation/test_all_notebooks_scripts.py
```

**Generate validation baselines:**
```bash
python tools/validation/output_to_validation.py
```

## Repository Architecture

- **`examples/`** — Domain-specific inversion examples (15 domains). Each directory is self-contained with notebooks, standalone scripts, and a `meta.yml` metadata file.
- **`tutorials/`** — Step-by-step CoFI tutorials (5 topics), structured similarly to examples.
- **`theory/`** — Markdown files explaining mathematical/geophysical theory behind examples.
- **`data/`** — Shared datasets used by examples.
- **`envs/`** — Conda (`environment.yml`) and pip (`requirements.txt`) dependency specifications.
- **`tools/`** — Developer utilities: example scaffolding, notebook execution, and validation suite.
- **`index.ipynb`** — Entry point notebook linking to all examples and tutorials.

## Example Directory Convention

Each example in `examples/<name>/` follows this pattern:
- `<name>.ipynb` — Primary interactive notebook (required)
- `<name>_<solver>.py` — Standalone Python scripts demonstrating specific solvers (optional)
- `<name>_lib.py` — Shared helper code imported by scripts/notebooks (optional, excluded from validation)
- `meta.yml` — Metadata: title, application domain hierarchy, description, and method classifications per file

## Validation System

The validation suite (`tools/validation/test_all_notebooks_scripts.py`) discovers and runs all `.py` files under `examples/`, excluding files ending in `lib.py` or starting with `_`. It compares output against baselines in `tools/validation/_validation_output/` using approximate numeric diff (tolerance 1e-3). Output goes to `tools/validation/_output/`.

## Commit Convention

This project uses [Angular-style](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#-git-commit-guidelines) commit messages: `feat:`, `fix:`, `docs:`, etc.

## Markdown Cell Note

In Jupyter notebook markdown cells, use `***` instead of `---` for horizontal rules to avoid pandoc YAML parsing errors.

## Change Log

### Rosenbrock + Neighpy Example (neighpy-examples branch)

Added a Neighbourhood Algorithm example applied to the Rosenbrock test function in `examples/metaheuristic_optimiser_tests/`:

- **`rosenbrock_neighpy.ipynb`** — Interactive notebook demonstrating all three CoFI neighpy tools (`neighpy`, `neighpyI`, `neighpyII`) on the log10-scaled Rosenbrock function. Includes Voronoi cell visualisations, appraisal sample scatter plots, and conditional distribution plots p(x|y=1) and p(y|x=1).
- **`rosenbrock_neighpy.py`** — Standalone script with argparse interface (`--output-dir`, `--show-plot`, `--save-plot`, `--show-summary`) following the existing script pattern.
- **`meta.yml`** — Updated with entries for both new files under the method path `CoFI -> Ensemble methods -> Direct search -> Monte Carlo -> Neighpy -> Neighbourhood Algorithm`.

Reference files used: `examples/linear_regression/linear_regression_neighpy.ipynb` (CoFI wrapper pattern) and `neighpy/examples/rosenbrock.ipynb` (Rosenbrock function definition, hyperparameters, visualisations).
