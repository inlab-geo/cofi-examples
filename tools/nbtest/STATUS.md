# nbtest status

Last updated: 2026-11-26. Update this file whenever work is done or a decision is made.

## Current state

- Design agreed (`DESIGN.md`). **Slice 1 (`run`, local mode) is committed as `b063399`. Slice 2A (state and local
  report) is committed as `c8034e1`; all Slice 2A documentation and code are pushed to `origin/nbtest`. Slice 2B
  environment/mode work has started with the required real-image preflight below.**
- Branch `nbtest`, created from `origin/main` at `d81ceb7`; pushed to `origin/nbtest` (tracking set). The maintainer's checkout lives in a
  separate git worktree on another branch and must not be switched.
- Nothing existing has been deleted or changed; old tools stay until the final cleanup step.

## Plan

1. **Run and report** — `./nbtest` wrapper, `mode`, `env`, `run` writing `*.nbtest.ipynb` with run/fail status,
   `state.json` + `REPORT.md` after every notebook, private kernel, image-mode isolation, `dependencies.txt`
   ordering and prerequisites, adding `caber_preprocessing.ipynb` to `examples/vtem_max/dependencies.txt`. Covers what
   `tools/run_notebooks/run_notebooks.py` does today.
2. **First references** — in the reference image: run all, bless all; validate again in the same image to find
   non-deterministic notebooks; fix with seeds or `nbtest-skip`.
3. **Validate** — number and figure comparison, annotations in staged notebooks; prototype on the smoke set, then
   all 39 notebooks.
4. **Publish** and remaining report details.
5. **Cleanup commit** (only after the user confirms): remove `tools/run_notebooks/`, `tools/post_update.sh`,
   `.github/workflows/test_all_notebooks.yml`; rewrite `tools/README.md`. Keep `tools/validation/`.
   Afterwards `inlab-containers/docs/how-it-works.md` needs updating — it names `run_notebooks.py all` as the
   image's integration test.

## Slice 2A: persistent state and local report (APPROVED 2026-11-26)

The smallest remaining local-mode foundation. It does not add environment fingerprints, image mode,
validation, references, blessing or publishing.

**In scope**

- `tools/nbtest/state/state.json` is local, versioned state written atomically. It contains a cumulative map of
  trusted staged notebooks and one latest-run record. `tools/nbtest/REPORT.md` is generated from that state.
- The latest run has one ordered notebook list containing only path, role (`target` or `prerequisite`) and
  status (`not_run`, `ok`, `failed`, `timeout`, `skipped` or `interrupted`). Supporting details and trusted
  original/staged hashes live in the cumulative notebook record; status is not duplicated there.
- Side effects belong only to the latest run and retain repository-relative path, change kind and Git-tracked
  flag. Hashes are used only during detection. Ignored untracked files, the expected staged copy and atomic
  temporary files are omitted. Detection is best effort: missing or inaccessible Git metadata records
  `side effects unavailable` but does not prevent execution.
- `run` preflights every selected staged copy before starting a kernel. A staged copy with no state hash
  (including Slice 1 output) or a changed hash is refused with a one-time upgrade explanation; `--force`
  permits replacement. The expected hash is checked again immediately before atomic replacement; a mid-run edit
  is left unchanged and recorded as `StagedConflict`. Forced replacement is printed but not persisted.
- `report` reads state and atomically rebuilds `REPORT.md` without importing execution dependencies. The report
  shows only the latest command. Interrupted and aborted runs retain the complete plan, with unreached entries
  shown as `not_run`.
- State loading parses JSON, checks schema version 1 and the core object containers. Corrupt or unsupported
  state stops before execution and tells the user to preserve state and staged notebooks. Additive fields do not
  bump the version and unknown notebook-record fields survive a rerun; a future breaking version must ship a
  migration rather than instruct users to delete state.
- Concurrent runs are unsupported and no lock is added. Atomic files prevent corruption; last writer wins and
  any staged/state mismatch is refused on the next run.
- A `try/finally` marks an unfinished run `aborted`; no internal exception diagnostics are stored in state.

**Acceptance groups (manual; no test framework)**

1. Normal fast run writes matching hashes and a report that `./nbtest report` reproduces byte-for-byte.
2. A Slice 1 staged file and then a manually edited trusted file are refused; `--force` replaces each. An edit
   made during execution is preserved as `StagedConflict`.
3. A targeted rerun preserves other trusted hashes; Ctrl-C records `interrupted` plus remaining `not_run`.
4. Nonignored side effects are listed, tracked changes warn, and ignored `__pycache__/` is absent.
5. Failure, timeout, corrupt state and unsupported state produce the agreed state/report or safe refusal.
6. `@smoke` retains Slice 1 outcomes and produces no unexpected tracked changes.

**Gates**

1. Specification approved. (done 2026-11-26)
2. Documentation committed before code.
3. Code written but not committed.
4. Acceptance output and before/after status shown to the maintainer.
5. Maintainer reviews the diff; only then may code be committed and pushed.

**Implementation and verification (2026-11-26)**

- Added versioned atomic state, latest-run reporting, `run --force`, staged SHA-256 preflight, side-effect
  detection and the dependency-free `report` command. User instructions and generated-file locations are in
  `README.md`.
- An existing Slice 1 staged notebook was refused without state. `--force` adopted it; a later manual edit was
  refused, and a second `--force` restored a matching saved/file hash.
- A failing notebook was recorded with its cell and error; `./nbtest report` reproduced `REPORT.md` byte for
  byte. `python -S ... report` also succeeded, confirming report generation does not import execution packages.
- Ctrl-C produced exit 130 with `interrupted` for the active notebook and `not_run` for two notebooks not
  reached. A later targeted run preserved another notebook's trusted hash.
- Ephemeral Git-repository checks covered created, modified and deleted side effects, the tracked-file warning,
  staged-file exclusion, and omission of ignored `__pycache__/`. Corrupt JSON and schema version 99 were safely
  refused. Syntax parsing and `git diff --check` pass.
- Review hardening checks: a mid-run edit survived and returned `StagedConflict`; the conditional atomic writer's
  returned digest matched the file; a no-write result preserved the older original/staged provenance pair and an
  unknown future field; removing `git` from `PATH` still ran the notebook and reported side effects unavailable;
  rapid double Ctrl-C exited 130 without a traceback and retained `interrupted` plus remaining `not_run` state.
- The local `cofidev` environment currently fails the nonlinear and linear regression examples at ArviZ's
  missing `plot_trace_dist`; its installed `pygimli` also fails on the internal module `pgcore`. nbtest records
  these as notebook failures as intended.
- Full non-Fedora smoke run (macOS 26.6.2 ARM64, `cofidev` Python 3.12.12, 2026-11-26): `./nbtest run @smoke`
  completed seven notebooks in 41.4 s — four `ok` (reduced-likelihood regression, X-ray tomography, FMM
  tomography and slime-mould) and the three environment failures above. All seven staged hashes matched state;
  no side effects or unavailable side-effect checks were recorded; Git status and the tracked diff were unchanged.
- Independent local environment run (same macOS ARM64 host, `cofCon` Python 3.10.18, cofi 0.2.10, nbclient
  0.10.2, NumPy 1.26.0, SciPy 1.15.2, Matplotlib 3.10.6, ArviZ 0.22.0): X-ray tomography was `ok` in 11.3 s
  with four figures and 19 outputs. Kernel/interpreter identity and the staged hash matched, no side effects were
  reported, Git status was unchanged, and `report` reproduced byte for byte. A separate Python 3.11 environment
  was rejected with exit 2 before execution because its standard library was broken; it changed no workspace
  state. Process start-time inspection confirmed the acceptance runs left no new kernels.

**Slice 2A is complete and pushed.** A forced timeout and an integrated notebook-created side-effect run remain
optional extensions to its acceptance matrix; they do not block Slice 2B.

## Slice 2B: mode and environment provenance (STARTED 2026-11-26)

No 2B code is written yet. Environment behavior must be measured before the wrapper and state interfaces are
changed.

**Required Fedora/image preflight — first gate**

1. On the Fedora maintainer machine, use the real reference `.sif`; do not substitute a built or downloaded
   image. Record Fedora, Apptainer and image paths/versions.
2. Inside that image, verify `command -v git` succeeds.
3. From inside the image, run `git -C <worktree> rev-parse --git-dir`, `--git-common-dir` and `ls-files`. Verify
   both resolved Git directories are visible. A worktree's `.git` is a pointer into the main checkout, and binding
   only the worktree can otherwise make tracked side-effect warnings silently unavailable.
4. If either Git check fails, stop before implementing image execution. Decide the minimum bind set and make
   image-mode startup fail clearly or make `bless` refuse a run whose side effects were unavailable; do not accept
   silent loss of the vtem_max tracked `.npy` warning.
5. After mode provenance is implemented, rerun `./nbtest run @smoke` on Fedora in the reference image and retain
   the raw outcomes, side-effect availability, hashes and Git before/after status.

**State rule for 2B**

- `begin_run` must receive the selected mode instead of hard-coding `local`.
- The notebook record gains `mode` and `environment_id` only when `staged_written` is true, alongside the paired
  original/staged hashes. A failed, interrupted or conflicted no-write attempt must preserve all four provenance
  values from the older staged copy.
- Future `bless` must require staged provenance from image mode and the selected reference environment; local or
  provenance-free staged copies are refused.

**First gate passed on Ceres (2026-11-26).** Host: Rocky Linux 8.10 with Apptainer 1.5.4. Image:
`/home/kitc/inlab-apptainer/inlab.sif`. Inside the image: Fedora 42 x86_64, Git 2.54.0, Python 3.13.13 and cofi
`0.2.12+19.gc0f3596`. From `/home/kitc/cofi-examples-nbtest`, Git resolved the linked gitdir and common directory
under `/home/kitc/cofi-examples/.git`; both were visible inside the image. The tracked vtem_max `.npy` probe
passed. Apptainer's default home binding therefore preserves tracked side-effect detection for the Ceres layout.

**Next implementation step:** explicit mode propagation and paired mode/environment provenance, followed by the
Fedora image `@smoke` run.

### Slice 2B.1: runtime context and provenance

**In scope**

- One runtime context per command: mode (`local` by default, `image` only when explicitly supplied), optional
  image path, complete environment document and stable short environment ID.
- Atomically write the latest document to `tools/nbtest/state/environment.json`. It records Python, OS/kernel/
  architecture, installed distributions, cofi and cofi-examples Git provenance where available, threading/BLAS
  details and matplotlib/FreeType details. Machine-specific paths are recorded but excluded from the stable ID.
- `begin_run` receives mode and environment ID; neither is hard-coded in state.
- A successfully written staged copy records `mode` and `environment_id` with its paired original/staged hashes
  in both state and notebook metadata. A no-write attempt preserves the older provenance quartet.
- `REPORT.md` shows the latest environment ID. `report` remains runnable without notebook-execution packages.
- Until the wrapper slice, `NBTEST_MODE=image` and `NBTEST_IMAGE_PATH=<sif>` are the private execution interface
  used only for the Ceres acceptance run; ordinary commands remain local by default.

**Out of scope for 2B.1:** remembered `.local.json`, public `mode`/`env` commands, Apptainer re-exec in the wrapper,
image hashing, validation, blessing and publishing.

**Acceptance**

1. Local run writes a reproducible environment document and matching IDs in latest-run and written-notebook state.
2. A simulated no-write result preserves older mode/environment/hash provenance and unknown future fields.
3. `report` regenerates byte for byte with site packages disabled.
4. Direct Ceres image run records `image`, the Fedora fingerprint and matching staged provenance; Git side-effect
   information remains available.

**Implementation status (2026-11-26): code written, uncommitted pending review**

- Added deterministic environment collection and atomic `state/environment.json`: Python, OS/kernel/architecture,
  installed distributions, cofi/repository Git provenance, deterministic thread settings, NumPy BLAS/LAPACK
  build details, threadpoolctl data where available, and matplotlib/FreeType details.
- Machine-specific paths are retained for diagnosis but excluded from the ID. Package and mode changes alter the
  ID. Raw installation URLs are not retained.
- `begin_run` receives mode/environment ID; written staged notebooks carry the same values in state and metadata.
  No-write simulation preserved the older hash/mode/environment quartet and unknown future fields.
- Local macOS ARM64 acceptance: repeated collection produced a stable ID and byte-identical document; a real
  notebook run had matching IDs in all four locations, matching staged hash, local Accelerate BLAS/LAPACK data,
  and a byte-reproducible report under `python -S`.
- Remaining gate: review and approval to commit/push the code, then direct Ceres image execution using the private
  `NBTEST_MODE=image`/`NBTEST_IMAGE_PATH` interface.

## Slice 1: `run` creates staged notebooks (DONE 2026-11-25, commit `b063399`)

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
  status, hash of the original. (The hash of the staged file as written moves to `state.json` in slice 2.)
- Terminal status per notebook: `ok`, `failed` (cell number + error), `timeout`, `skipped (missing: <pkg>)`
  (only for modules listed in `tools/nbtest/optional_packages.txt` that are not installed); summary line; exit
  code 0 / 1 (failed or timeout) / 2 (usage or target error) / 130 (Ctrl-C). Default timeout 3 h per notebook.
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
   `three_survey_lines_latin_hypercube_sampling`, `three_survey_lines_surrogate_model_creation` and
   `caber_preprocessing` (flat list, accepted 2026-11-25) as prerequisites before the target (all `skipped (missing: …)` locally, since PyP223 and smt are not installed).
   A typo'd `dependencies.txt` entry and a path outside the scope (e.g. `index.ipynb`) each give a clear error.

**Gates** (each needs the maintainer's OK before the next):

1. This spec approved. (done)
2. Code written. (done)
3. Smoke run: raw output and before/after diff shown to the maintainer. (done)
4. Independent review by a fresh agent given the code and this spec, not the author's conclusions. (done; all
   findings fixed and re-verified, see below)
5. Maintainer reviews the diff; only then are code changes committed. (done: approved, committed and pushed)

**Limits:** image mode cannot be verified on this machine (no `.sif`); tolerances are unproven until slice 3.

**Verification results (2026-11-25, local mode, `inlab` venv, Python 3.14.10):**

- `./nbtest run @smoke`: 6 ok, `pygimli_ert` skipped (missing: pygimli), ~35 s, exit 0. Before/after snapshot:
  only staged `*.nbtest.ipynb` files and `examples/xray_tomography/__pycache__/` appear; no tracked file changes.
- Kernel Python recorded in every staged notebook = the `inlab` venv. Figure counts equal the committed
  notebooks, except `slime_mould_algorithm_demo`: its committed cell 8 holds 5 stale figures from out-of-order
  runs, while the code shows 1; nbtest's 1 is correct.
- Timeout (`--timeout 1`, `--timeout 3`), single and triple Ctrl-C (exit 130, no traceback), dead kernel: no
  leftover kernels, no temp files, existing staged copy untouched on Ctrl-C.
- Old runner as oracle (scratch copy): agrees after explaining slime_mould (seaborn was installed in the venv
  between the two runs).
- Synthetic notebooks: `import numpyy` and a missing local helper → failed; `import cofi.xyz` → failed;
  `import pygimli` (listed, not installed) → skipped; dependencies.txt typo → error, exit 2.

**Independent review (fresh agent) findings and resolution:**

1. Any unresolvable import was "skipped", exit 0 → only `optional_packages.txt` entries can be skipped (decided).
2. Acceptance test 7 had 2 prerequisites; the flat list gives 3 → test updated (flat list accepted).
3. Staged files were created 0600 → staged copy gets the original's permissions.
4. dependencies.txt typos (non-`.ipynb` lines) were ignored → every non-blank, non-comment line must name an
   existing notebook.
5. Second Ctrl-C printed a traceback; Ctrl-C between notebooks not handled → handled.
6. `staged_sha256` could not be the hash of the file itself → removed; moves to `state.json` (decided).
7. Dead-kernel log showed only noise → known noise filtered; section omitted when empty.
8. Kernel not stopping within the grace period → kernel killed, no staged copy written, reported as failed.
9. Cancellation during kernel start-up relied on exit handlers → kernel process killed directly after each
   notebook if still alive.
10. "39 vs 40 notebooks" → explained: `variational_inference` exists only on `gaussian-vi`.

**Notes for later slices:** ipykernel 7.3 prints "Kernel is running over TCP without encryption" per kernel
(now hidden in the kernel log). `examples/xray_tomography` leaves a `__pycache__/` folder (ignored by git).

## Smoke-test set

For development, use these (`tools/nbtest/smoke.txt`, target `@smoke`). Run time is not a concern.

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

- `dependencies.txt` lists `three_survey_lines_latin_hypercube_sampling.ipynb`,
  `three_survey_lines_surrogate_model_creation.ipynb`, then `caber_preprocessing.ipynb`. The old runner runs
  listed notebooks first, then the rest in `os.listdir` order (arbitrary).
- Chain: latin_hypercube_sampling writes `three_survey_lines_lhs.npy` (**tracked in git**) →
  surrogate_model_creation reads it and writes `kriging_surrogate_model.pkl` (untracked, generated) →
  `three_survey_lines_smt_bayesbay`, `_smt_emcee`, `_smt_neighpy` read the `.pkl`.
- `caber_preprocessing.ipynb` writes `caber_survey.npy` (**tracked**), which `caber_inversion.ipynb` reads.

Agreed handling (also in `DESIGN.md`):

1. Order within a folder: `dependencies.txt` entries in their order, then the remaining notebooks alphabetically
   (deterministic, unlike `os.listdir`). Always sequential within a folder.
2. Selecting a single notebook in such a folder first runs the listed prerequisites (those before it, if it is
   itself listed), so `run examples/vtem_max/three_survey_lines_smt_emcee.ipynb` works on a fresh clone.
   Prerequisites appear in the report as prerequisites.
3. Execute in place (consistent with editing the staged copy in Jupyter). Snapshot the folder before and after;
   report created/modified files and warn when a tracked file was modified (e.g. the LHS `.npy`, which is random
   unless seeded).
4. `caber_preprocessing.ipynb` was added to `examples/vtem_max/dependencies.txt` in slice 1.

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
