# cofi-beans

CoFI use case notebooks

These are the notebooks illustrating the three use cases of CoFI:

- **educator** — A tutorial progressing from polynomial fitting to change-point models using Bayesian inference. Demonstrates CoFI's unified interface for linear solvers, optimisation, and MCMC sampling, applied to sea level data.
- **researcher** — Explores surface wave tomography with CoFI, comparing different regularisation options and ensemble methods.
- **developer** — Shows how developers can integrate new algorithms into CoFI, using receiver function analysis as an example.

## Rendered notebooks

### Jupyter notebooks

Full builds also export Jupyter notebooks with outputs to `rendered/`:

| Notebook | Description |
|---|---|
| [educator](rendered/educator.ipynb) | Polynomial fitting and change-point models |
| [researcher](rendered/researcher.ipynb) | Surface wave tomography |
| [developer](rendered/developer.ipynb) | Extending CoFI with custom solvers |

These are generated with `--include-outputs` (outputs embedded) and require `nbformat`. They are not produced by `--fast` builds.

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

Use `--fast` for a quick development build (1% rays, minimal MCMC iterations):

```bash
./build.sh --fast
./build.sh --fast researcher.py
```

Run `./build.sh --help` for full usage details.

This exports HTML pages with outputs to `rendered/`. Full builds (without `--fast`) also generate Jupyter notebooks (`.ipynb`) with outputs.
