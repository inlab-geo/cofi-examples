# nbtest status

Last updated: 2026-11-25. Update this file whenever work is done or a decision is made.

## Current state

- Design agreed (`DESIGN.md`). **No code written yet.**
- Branch `nbtest`, created from `origin/main` at `d81ceb7`. Not pushed. The maintainer's checkout lives in a
  separate git worktree: `../cofi-examples` is on another branch (`gaussian-vi`) and must not be switched.
- Nothing existing has been deleted or changed; old tools stay until the final cleanup step.

## Plan

1. **Run and report** — `./nbtest` wrapper, `mode`, `env`, `run` writing `*.nbtest.ipynb` with run/fail status,
   `state.json` + `REPORT.md` after every notebook, private kernel, image-mode isolation, `dependencies.txt`
   ordering and prerequisites, adding `caber_preprocessing.ipynb` to `examples/vtem_max/dependencies.txt`. Covers what
   `tools/run_notebooks/run_notebooks.py` does today.
2. **First references** — in the reference image: run all, bless all; validate again in the same image to find
   non-deterministic notebooks; fix with seeds or `nbtest-skip`.
3. **Validate** — number and figure comparison, annotations in staged notebooks; prototype on the smoke set, then
   all 40 notebooks.
4. **Publish** and remaining report details.
5. **Cleanup commit** (only after the user confirms): remove `tools/run_notebooks/`, `tools/post_update.sh`,
   `.github/workflows/test_all_notebooks.yml`; rewrite `tools/README.md`. Keep `tools/validation/`.
   Afterwards `inlab-containers/docs/how-it-works.md` needs updating — it names `run_notebooks.py all` as the
   image's integration test.

## Slice 1: `run` creates staged notebooks (spec approved 2026-11-25; gate 2, code, next)

The first self-contained piece of step 1. Local mode only.

**In scope**

- Target selection: `all`, a folder, notebook paths, `@smoke` (`tools/nbtest/smoke.txt`). Discovery exactly one
  folder below `examples/` and `tutorials/`, skipping `*.nbtest.ipynb`.
- `dependencies.txt`: fixed order and automatic prerequisites; add `caber_preprocessing.ipynb` to
  `examples/vtem_max/dependencies.txt`.
- Execution with `nbclient` and a private, uniquely named kernelspec using nbtest's own Python; working directory =
  the notebook's folder; per-cell timeout plus per-notebook wall-clock watchdog; thread and hash-seed settings;
  **no `MPLBACKEND`**.
- Writes `<name>.nbtest.ipynb` next to the original, atomically; adds `*.nbtest.ipynb` to `.gitignore`.
- Stamps `metadata.nbtest` in the staged notebook: start time, duration, Python (`sys.prefix`, `sys.executable`),
  status, hash of the original, hash of the staged file as written (later used by the edit guards).
- Terminal status per notebook: `ok`, `failed` (cell number + error), `timeout`, `skipped (missing: <pkg>)`
  (top-level module not installed); summary line; exit code 0/1.
- `./nbtest` wrapper as a thin shell around `python -m cofi_nbtest run`.
- Errors (decided 2026-11-25): a `dependencies.txt` entry naming a missing notebook stops with an error; paths
  outside notebooks one folder below `examples/` and `tutorials/` are refused with an error.
- **No test framework for nbtest itself** (maintainer decision 2026-11-25): no pytest suite; nbtest is checked
  through the acceptance tests below.

**Out of scope for slice 1:** modes/image, `environment.json`, `state.json`/`REPORT.md`, validate, bless, publish,
annotations, edit guards, reporting files changed by a run. Known risk: slice 1 overwrites staged copies without
checking for edits (nobody edits them yet).

**Acceptance tests**

1. `./nbtest run @smoke`: six notebooks `ok`, or `failed` with a clear error; `pygimli_ert` reports
   `skipped (missing: pygimli)`; seven staged files written.
2. Side effects: snapshot `git status --porcelain --ignored` plus a file listing before and after; the difference
   is exactly the staged files (plus `__pycache__`). No tracked file changes.
3. Every staged notebook's `metadata.nbtest` names the `inlab` venv Python (`sys.prefix`).
4. Figures: each staged notebook has as many `image/png` outputs as its code produces (checked against a run
   without nbtest).
5. Ctrl-C during a run and a forced timeout: clean stop, no half-written staged file, no leftover `ipykernel`
   processes.
6. **Oracle:** in a scratch copy of the worktree (never the worktree or the main checkout, since the old runner
   overwrites notebooks in place), run `tools/run_notebooks/run_notebooks.py` on the smoke notebooks and compare
   pass/fail per notebook with nbtest. Every disagreement must be explained.
7. Ordering and prerequisites: `./nbtest run examples/vtem_max/three_survey_lines_smt_emcee.ipynb` lists
   `three_survey_lines_latin_hypercube_sampling` then `three_survey_lines_surrogate_model_creation` as
   prerequisites before the target (all `skipped (missing: …)` locally, since PyP223 and smt are not installed).
   A typo'd `dependencies.txt` entry and a path outside the scope (e.g. `index.ipynb`) each give a clear error.

**Gates** (each needs the maintainer's OK before the next):

1. This spec approved. (done)
2. Code written.
3. Smoke run: raw output and before/after diff shown to the maintainer.
4. Independent review by a fresh agent given the code and this spec, not the author's conclusions.
5. Maintainer reviews the diff; only then are code changes committed.

**Limits:** image mode cannot be verified on this machine (no `.sif`); tolerances are unproven until slice 3.

## Smoke-test set

For development, use these (planned file `tools/nbtest/smoke.txt`, target `@smoke`). Run time is not a concern.

| Notebook | Why |
| --- | --- |
| `examples/linear_regression/linear_regression.ipynb` | main case; several methods incl. emcee sampler |
| `examples/linear_regression/linear_regression_reduced_likelihood.ipynb` | deterministic, 8 figures |
| `examples/nonlinear_curve_fitting/nonlinear_curve_fitting.ipynb` | very fast; emcee |
| `examples/xray_tomography/xray_tomography.ipynb` | deterministic, image-type figures, no extra deps |
| `examples/fmm_tomography/fmm_tomography.ipynb` | compiled solver installed from git (pyfm2d) |
| `examples/metaheuristic_optimiser_tests/slime_mould_algorithm_demo.ipynb` | nbformat 4.4, no cell ids; mealpy |
| `examples/pygimli_ert/pygimli_ert.ipynb` | missing-package case locally (`skipped (missing: pygimli)`) |

Later, image mode only: the vtem_max pair `three_survey_lines_latin_hypercube_sampling` →
`three_survey_lines_surrogate_model_creation` to exercise `dependencies.txt`.

## Open questions

1. **Reference figure resolution** — 400 px downscaled vs full resolution (~40 MB more). Asked as a comment in the
   shared design doc; not answered yet.

## `dependencies.txt` (agreed 2026-11-25)

Facts (checked 2026-11-25, only `examples/vtem_max/` has a `dependencies.txt`):

- `dependencies.txt` lists `three_survey_lines_latin_hypercube_sampling.ipynb` then
  `three_survey_lines_surrogate_model_creation.ipynb`. The old runner runs listed notebooks first, then the rest
  in `os.listdir` order (arbitrary).
- Chain: latin_hypercube_sampling writes `three_survey_lines_lhs.npy` (**tracked in git**) →
  surrogate_model_creation reads it and writes `kriging_surrogate_model.pkl` (untracked, generated) →
  `three_survey_lines_smt_bayesbay`, `_smt_emcee`, `_smt_neighpy` read the `.pkl`.
- Not listed: `caber_preprocessing.ipynb` writes `caber_survey.npy` (**tracked**), which `caber_inversion.ipynb`
  reads. It works today only because the `.npy` is committed.

Agreed handling (also in `DESIGN.md`):

1. Order within a folder: `dependencies.txt` entries in their order, then the remaining notebooks alphabetically
   (deterministic, unlike `os.listdir`). Always sequential within a folder.
2. Selecting a single notebook in such a folder first runs the listed prerequisites (those before it, if it is
   itself listed), so `run examples/vtem_max/three_survey_lines_smt_emcee.ipynb` works on a fresh clone.
   Prerequisites appear in the report as prerequisites.
3. Execute in place (consistent with editing the staged copy in Jupyter). Snapshot the folder before and after;
   report created/modified files and warn when a tracked file was modified (e.g. the LHS `.npy`, which is random
   unless seeded).
4. Add `caber_preprocessing.ipynb` to `examples/vtem_max/dependencies.txt` (a one-line data change; to do as
   part of step 1, not done yet).

## Environment notes

- Maintainer machine (2026-11-25): Fedora 45, Apptainer 1.5.5. The host virtualenv `~/.virtualenvs/inlab`
  (Python 3.14) has cofi, emcee, arviz, neighpy, bayesbay, pyfm2d, mealpy, torch, seislib, cartopy, papermill,
  nbformat, nbclient, ipykernel, threadpoolctl — **not** pygimli, pyrf96, pysurf96, pyhk, PyP223, smt.
  Useful for testing local mode and the missing-package path.
- No `.sif` image is available on this machine yet; ask the user for a path before testing image mode. Do not
  download or build images unprompted.
- All 40 notebooks have `kernelspec.name == "python3"`; 15 are nbformat 4.4 without cell ids (all of vtem_max,
  `variational_inference`, `receiver_function_neighpy`, `slime_mould_algorithm_demo`, `rosenbrock_neighpy`).
- Committed notebook outputs come from mixed environments (Python 3.9–3.14) and some are unexecuted; they are
  not used as references.
