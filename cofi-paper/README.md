# cofi-beans

CoFI use case notebooks

These are the notebooks illustrating the three use cases of CoFI:

- **educator** — A tutorial progressing from polynomial fitting to change-point models using Bayesian inference. Demonstrates CoFI's unified interface for linear solvers, optimisation, and MCMC sampling, applied to sea level data.
- **practitioner** — Explores surface wave tomography with CoFI, comparing different regularisation options and ensemble methods.
- **developer** — Shows how developers can integrate new algorithms into CoFI, using receiver function analysis as an example.

## Tutorial versions

Curated Jupyter versions of these notebooks are available as tutorials in [`tutorials/`](../tutorials/). They do not need marimo and read their data from the shared top-level `data/` folder. Each tutorial folder also holds a marimo copy of the corresponding notebook, with only its data and figure paths adapted.

| Use case | Tutorial | Marimo copy |
|---|---|---|
| educator | [Curve fitting with CoFI](../tutorials/sea_level_change/sea_level_change.ipynb) | [`sea_level_change.py`](../tutorials/sea_level_change/sea_level_change.py) |
| practitioner | [Ambient noise tomography with CoFI](../tutorials/surface_wave_tomography/surface_wave_tomography.ipynb) | [`surface_wave_tomography.py`](../tutorials/surface_wave_tomography/surface_wave_tomography.py) |
| developer | [Extending CoFI with a new inference tool](../tutorials/extending_cofi_new_solver/extending_cofi_new_solver.ipynb) | [`extending_cofi_new_solver.py`](../tutorials/extending_cofi_new_solver/extending_cofi_new_solver.py) |

The Gaussian variational inference extension of the practitioner notebook lives in the tomography tutorial folder, as `surface_wave_tomography_variational_inference_extension` (`.ipynb` and `.py`).

The notebooks in this folder remain the source for the paper. The tutorials are independent copies, so changes made here are not propagated to them automatically.

## Rendered notebooks

### Jupyter notebooks

Full builds also export Jupyter notebooks with outputs to `rendered/`:

| Notebook | Description |
|---|---|
| [educator](rendered/educator.ipynb) | Polynomial fitting and change-point models |
| [practitioner](rendered/practitioner.ipynb) | Surface wave tomography |
| [developer](rendered/developer.ipynb) | Extending CoFI with custom solvers |

These are generated with `--include-outputs` (outputs embedded) and require `nbformat`. This export pass is skipped by `--fast` builds. Each notebook renders in a fixed mode: educator and developer render in full mode; practitioner renders in fast mode (edit `run_mode` in `notebooks/practitioner.py` to render it in full mode).

## Development

- `notebooks/` — marimo source notebooks (`.py`) for development
- `rendered/` — exported HTML pages and Jupyter notebooks with outputs
- `data/` — input datasets
- `build.sh` — renders all notebooks (or a single one) to `rendered/`

For local viewing, start a small static server:

```bash
python3 -m http.server --directory rendered 8000
```

Then open `http://localhost:8000`.

## Using Marimo

### Installation

```bash
pip install marimo
```

### Running notebooks

```bash
marimo edit notebooks/educator.py
```

### Running headless (remote server)

To run marimo on a remote server and access it from your local browser, start
it in headless mode and forward the port over SSH:

```bash
# On the remote server
marimo edit notebooks/educator.py --headless --port 2718
```

```bash
# On your local machine (in a separate terminal)
ssh -L 2718:localhost:2718 user@remote-host
```

Then open `http://localhost:2718` in your local browser.

### Building rendered notebooks

Build all notebooks:

```bash
./build.sh
```

Build a single notebook:

```bash
./build.sh educator.py
```

Use `--fast` to skip the slower Jupyter (`.ipynb`) export pass:

```bash
./build.sh --fast
./build.sh --fast practitioner.py
```

Run `./build.sh --help` for full usage details.

This exports HTML pages with outputs to `rendered/`. Builds without `--fast` also generate Jupyter notebooks (`.ipynb`) with outputs. Educator and developer always render in full mode; practitioner always renders in fast mode via `build.sh` — full mode is 350,000 MCMC iterations and can take multiple hours, so it is never run automatically.
