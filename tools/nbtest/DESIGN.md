# nbtest design

nbtest is a local, terminal-driven tool that runs the cofi-examples notebooks and checks whether they still
reproduce their reference numbers and figures. It replaces `tools/run_notebooks/`.

Status: design agreed 2026-11-25, nothing implemented yet. See `STATUS.md` for progress and open items.
A shared copy of this design (with a data-flow diagram) is at
https://claude.ai/artifact/MPVJxUG6bSMCocTL1Yy6CN — this file is the source of truth if they differ.

## Use cases

- **Maintainers** run all notebooks inside an Apptainer image (the reference environment), validate against the
  references, and accept new references (`bless`) when a change is intended.
- **An external user ("Bob")** has installed cofi locally, clones cofi-examples and wants one command that runs
  every notebook and tells him whether he gets the reference results. He has no Apptainer image and different
  package versions.
- Results most often stop matching because a Python package was updated, so every run records the full
  environment and the report shows what changed.

## Commands

All commands go through the `./nbtest` wrapper at the repo root. Targets: `all`, an example folder, one or more
`.ipynb` paths, or `@<list>` (a file of paths, e.g. `tools/nbtest/smoke.txt`).

| Command | Reads | Writes | Notes |
| --- | --- | --- | --- |
| `run` | original notebooks | `<name>.nbtest.ipynb` next to each original | Never modifies the original notebook. Refuses to overwrite a staged notebook with unpublished edits unless `--force`. Updates `state.json` and `REPORT.md` after every notebook. |
| `validate` | staged notebooks, `references/` | annotations in staged notebooks, `diffs/`, `state.json`, `REPORT.md` | `--run` runs first. `--strict` makes `changed` a failure in local mode. |
| `bless` | staged notebooks | `references/` | Image mode only. Refuses notebooks whose last run failed. Never commits. |
| `publish` | staged notebooks | original notebooks | Both modes. Strips nbtest annotations and tags, stamps mode + environment fingerprint id into notebook metadata, copies over the original. Refuses if the original changed since `run`. Never commits. |
| `report` | `state.json` | `REPORT.md` | Rebuilds the report without running anything. |
| `mode` | `.local.json` | `.local.json` | `mode local`, `mode image [path.sif]`, `mode` shows the current mode. |
| `env` | current environment | stdout | `--diff` compares with `references/environment.json`. |

Typical use:

```console
$ ./nbtest validate --run all                      # Bob, or a maintainer in local mode
$ ./nbtest mode image ~/images/inlab.sif           # switch to the reference image (remembered)
$ ./nbtest validate --run examples/fmm_tomography
$ ./nbtest bless examples/fmm_tomography
$ ./nbtest publish examples/fmm_tomography
```

## Modes

- **local** (default): runs in the Python environment nbtest is started from. Loose tolerances. `bless` refused.
  A notebook whose required top-level package is not installed is `skipped (missing: …)`.
- **image**: every command runs inside an Apptainer image via `apptainer exec`. Strict tolerances. `bless` allowed.
- `mode image <path.sif>` stores the path in `tools/nbtest/.local.json` (gitignored, per clone). `mode image`
  without a path reuses it; if none is stored or the file is gone it asks once; with no terminal it errors with
  instructions.
- Every command prints a banner, e.g. `nbtest mode: image (/path/inlab.sif)` or
  `nbtest mode: local (Python 3.14.10, cofi 0.2.12)`.
- The reference image is whatever `.sif` was used to bless; nbtest records its name and sha256 (computed at bless
  time only) and does not care how it was built. Building images is outside nbtest.
- Mode handling, the prompt and `apptainer exec` live in the shell wrapper, so nbtest never depends on the host
  Python in image mode. Everything else runs as `python -m cofi_nbtest` inside the chosen environment.

## Staged notebooks

- `run` writes `examples/<dir>/<name>.nbtest.ipynb` next to the original. The pattern `*.nbtest.ipynb` is
  gitignored and skipped by notebook discovery.
- Notebooks execute with their own folder as working directory (relative data paths keep working), the same
  folder a user gets when opening the staged copy in Jupyter.
- `run` stores hashes of the staged notebook it wrote and of the original. If the staged copy was edited since,
  `run` refuses to overwrite it (`--force` overrides). If the original changed since, `publish` refuses.

## References

Committed in `tools/nbtest/references/`, mirroring repo paths, one folder per notebook (name without `.ipynb`):

```
references/
├── environment.json                         # environment the references were blessed in
└── examples/fmm_tomography/fmm_tomography/
    ├── values.json                          # numbers per cell
    ├── cells.json                           # per code cell: id, position, source hash, first line,
    │                                        #   nearest markdown heading above, output types, errors, warnings
    └── fig_<cell>_<n>.png                   # figures, downscaled (about 400 px wide; open question)
```

- Compact extracts rather than full notebook copies: ~5–8 MB for all ~200 figures versus 48 MB, readable diffs.
- The published notebooks on GitHub/Colab still show full outputs; they are independent of the references.

## Environment fingerprint (`environment.json`)

Written for every run (`tools/nbtest/state/`, local) and committed with the references on bless. Contains:

- Python version and interpreter path; OS release, kernel, architecture, CPU
- container: Apptainer or not, `.sif` path, sha256 (bless time only)
- every installed distribution with version; for git installs the commit from `direct_url.json`
  (`vcs_info.commit_id`); for editable/local installs the live `git rev-parse HEAD` and dirty flag of the source
  (dist-info metadata can be stale for editable installs — observed with cofi)
- BLAS/LAPACK backend and thread counts in effect (`threadpoolctl` if available); relevant env vars
- matplotlib version, backend, FreeType version, default font
- cofi-examples commit + dirty flag; cofi version
- a short fingerprint id over the version-sensitive parts, e.g. `env-3f9a1c`

`env --diff` and the top of `REPORT.md` show only what differs from the reference.

## What is compared

Everything a notebook produces, by default: numbers in outputs, figures, errors, which cells produce output.

- **Cell matching:** by source hash and position; cell ids are only a hint, because nbformat assigns random ids
  when reading the 15 notebooks still in nbformat 4.4. Cells whose code changed are still compared and labelled
  "edited cell"; their differences are shown but do not count as failures in image mode.
- **Opt-out:** Jupyter cell tag `nbtest-skip` (in `metadata.tags`): the cell runs, its outputs are not compared.
  The report counts skipped cells.
- **Numbers:** text outputs are tokenised into words and numbers and aligned against the reference with a diff
  algorithm, not by position. Numbers inside names are ignored (`np.float64`, `x2`, `v1.0`). "Text changed" is
  reported separately from "value changed".
- **Automatic exclusions** (machine-dependent): explicit timing output only (`Wall time`, `CPU times`, `%timeit`,
  `elapsed`, `took … s`, `it/s`, tqdm and sampler progress bars); dates and times; memory addresses, paths, host
  and user names, process ids; thread/core counts and memory sizes; the watermark cell's version output;
  stderr (warnings are listed as new/gone, not compared). **Never exclude numbers by unit alone** — seconds are
  physical results in travel-time and receiver-function examples.
- **Figures:** a change of image size is reported as a difference in its own right; images are resized to a common
  size and compared with matplotlib's `compare_images` (RMS on 0–255).
- **Tolerances:** fixed defaults per mode, no per-notebook configuration unless real runs show a need.

| | Image mode | Local mode |
| --- | --- | --- |
| Numbers `rtol` / `atol` (`numpy.isclose` rule) | 1e-9 / 1e-12 | 1e-3 / 1e-8 |
| Figures, RMS | 0.5 | 10 |

## Statuses

| What happened | Image mode | Local mode |
| --- | --- | --- |
| Notebook raised an error | failed | failed |
| Required package not installed | failed | skipped (missing: …) |
| Figure or output added or removed | failed (structure changed) | failed (structure changed) |
| Number or figure beyond tolerance | changed, counts as failure | changed, warning |
| New warning on stderr | reported | reported |
| No reference | no-ref | no-ref |
| Everything within tolerance | pass | pass |

- A package is "missing" only if its top-level module is not installed at all (`importlib.util.find_spec` in the
  kernel's Python). `ModuleNotFoundError` for a submodule of an installed package (e.g. `cofi.xyz`) is a failure.
- `validate` exits 0 if nothing failed, 1 otherwise.
- Sampler notebooks get no special handling. After the first bless, validating again in the same image reveals
  non-deterministic notebooks; fix with a seed or `nbtest-skip`.

## Reviewing and fixing failures

`validate` writes into the staged notebook:

- a summary cell at the top listing problem cells;
- a markdown cell directly above each problem cell: reference vs new values, figure RMS, links to reference and
  difference images in `diffs/`; JupyterLab alert styling (red = failed, amber = changed);
- tags: `nbtest-annotation` on inserted cells, `nbtest-failed` / `nbtest-changed` on the problem cell.

`REPORT.md` names the same cells (number, section heading, first line of code) and links to the staged notebook.
Workflow: open `<name>.nbtest.ipynb`, fix and re-run cells, `./nbtest publish <nb>` (strips annotations, copies
over the original), `./nbtest validate --run <nb>`.

## Running notebooks

- Execution uses `nbclient` directly (papermill is not needed). nbtest's own requirements (nbclient, nbformat,
  matplotlib, Pillow, optionally threadpoolctl) are listed in the README; missing ones give a clear message.
- **Private kernel:** all notebooks request kernel `python3`, which Jupyter resolves through user kernelspecs and
  which may point at any Python. nbtest registers a temporary kernelspec in its own `JUPYTER_PATH` whose argv is
  `[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]` and always uses it; it checks the
  kernel's `sys.executable` matches `environment.json`.
- **Image-mode isolation:** `apptainer exec --cleanenv`, `PYTHONNOUSERSITE=1`, a private `MPLCONFIGDIR`, the repo
  bind-mounted, and an entry script that puts nbtest on `sys.path` (do not override `PYTHONPATH`; images may set
  it, e.g. for pygimli). The wrapper checks it is at the repo root.
- **Determinism (both modes):** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 1,
  `MPLBACKEND=Agg`, `PYTHONHASHSEED=0`.
- **Discovery:** `.ipynb` exactly one folder below `examples/` and `tutorials/` (excludes `archived/`), skipping
  `*.nbtest.ipynb`.
- **Within a folder notebooks run sequentially**; different folders may run in parallel. Order: the entries of
  the folder's `dependencies.txt` (if any) in their order, then the remaining notebooks alphabetically.
- **Prerequisites:** selecting a single notebook in a folder with `dependencies.txt` first runs the listed
  prerequisites (those listed before it, if it is itself listed), so it works on a fresh clone where generated
  files (e.g. vtem_max's `kriging_surrogate_model.pkl`) do not exist yet. They are shown as prerequisites in the
  report.
- **In place:** notebooks execute in their own folder, as when the staged copy is opened in Jupyter. The folder is
  snapshotted before and after; files a run created or modified are reported, with a warning when a tracked file
  changed (e.g. vtem_max's `three_survey_lines_lhs.npy`, which differs on every run unless seeded).
  See `STATUS.md` for the vtem_max file chain.
- Per-notebook timeout. `state.json` is written atomically (temp file + rename) after every notebook, including
  failures and Ctrl-C.

## Files and layout

```
cofi-examples/
├── nbtest                          # wrapper: mode handling, apptainer exec, then python -m cofi_nbtest
├── examples/<dir>/<name>.ipynb     # original, committed
├── examples/<dir>/<name>.nbtest.ipynb   # staged copy (gitignored)
└── tools/nbtest/
    ├── DESIGN.md  STATUS.md  AGENTS.md  README.md
    ├── smoke.txt                   # smoke-test notebook list
    ├── cofi_nbtest/                # cli, discover, runner, compare, annotate, state, report, env
    ├── references/                 # committed
    ├── state/                      # gitignored: state.json, environment.json of the last run
    ├── REPORT.md                   # gitignored, generated from state.json
    ├── diffs/                      # gitignored: reference / new / difference images
    └── .local.json                 # gitignored: mode and image path
```

The internal package is `cofi_nbtest` because the name `nbtest` is taken on PyPI (an unrelated project); the
command and tag names are still `nbtest`.

## Scope

In scope: notebooks one folder below `examples/` and `tutorials/` (40 on 2026-11-25).

Out of scope: CI (run by hand); Docker (Apptainer only); building images and the `inlab-apptainer` definitions;
`cofi-paper` marimo notebooks (own `build.sh`); standalone example `.py` scripts (`tools/validation/` keeps
testing them); per-notebook tolerances, statistical sampler checks, named `record(...)` values (later, if needed).

## Decision log

| Decision | Reason |
| --- | --- |
| Local, terminal tool; no CI | The team cannot afford CI time for these notebooks. |
| Apptainer only, no Docker | The team uses Apptainer; image choice is just a `.sif` path. |
| Reference image = the `.sif` used to bless, recorded by name + sha256 | Keeps nbtest independent of how images are built; `.def` issues are handled elsewhere. |
| `nbtest mode local|image`, remembered in `.local.json` | Local by default for Bob; image is handy for maintainers; banner prevents surprises. |
| References committed in git, separate from the notebooks | Bob must be able to validate after cloning; notebooks are actively edited, so they cannot be the reference. |
| Compact extracts instead of full notebook copies | Repo size (already ~464 MB packed) and readable diffs. |
| `environment.json` is the only provenance | Mismatches are almost always caused by updated Python packages; it records exactly that. |
| Everything compared; `nbtest-skip` cell tag to opt out | Opt-out lives in the notebook, not a config file. |
| Fixed tolerances per mode, no config | Avoid over-engineering until real runs show a need. |
| `changed` is a warning in local mode, failure in image mode | Differences are expected with other package versions. |
| Figure/output added or removed always fails | Usually a silently broken plot cell. |
| Staged copy next to the original, annotated in place | Users review and fix directly in the staged notebook; relative paths keep working. |
| `publish` in both modes, stamped | Does not block anyone; makes the origin of published outputs visible. |
| No special sampler handling | A second run in the same image exposes non-determinism; fix with seeds. |
| Internal package `cofi_nbtest` | `nbtest` is taken on PyPI. |
| Review fixes 3–10 adopted (2026-11-25) | Tokenised number alignment, explicit timing exclusions only, source-hash cell matching, compare edited cells, publish guard, sequential per folder, nbclient + clear deps, figure-size handling. |
| `dependencies.txt`: listed first then alphabetical, sequential per folder; prerequisites auto-run for single-notebook targets; run in place and report created/modified files; add `caber_preprocessing.ipynb` to vtem_max's list (2026-11-25) | vtem_max notebooks pass files to each other; a fresh clone lacks generated files; in-place matches editing the staged copy; `caber_inversion` only worked because its input is committed. |
| Not adopted (2026-11-25): `env --requirements` for Bob; `bless` refusing a different image sha256 | Considered in review; the user did not take them up. Do not re-add without asking. |
