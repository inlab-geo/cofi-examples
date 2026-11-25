# Instructions for AI agents working on nbtest

Read `DESIGN.md` and `STATUS.md` in this folder before doing anything. They hold the agreed design, the
decision log and the current progress.

## Working rules

- **Plan before building.** The maintainer discusses design first. Do not write code, create files or start the
  next step until they explicitly say to build. Agreement with a design point is not a go-ahead.
- **Respect the decision log** in `DESIGN.md`. Do not re-propose items marked "not adopted" or out of scope
  unless the maintainer raises them.
- **Keep it simple.** The maintainer prefers minimal mechanisms; add configuration or features only when real
  runs show a need.
- **Branch and worktree.** Work happens on the `nbtest` branch, in its own git worktree. Never switch the
  maintainer's main checkout (another branch is checked out there).
- **Commit every Markdown file you create or change** in the nbtest work (e.g. `DESIGN.md`, `STATUS.md`) to the
  `nbtest` branch straight away; the maintainer wants all docs committed there. Otherwise never commit code or
  push without being asked. Never delete existing tools (`tools/run_notebooks/`,
  `tools/validation/`, workflows) before the final cleanup step is explicitly approved.
- **Images.** Apptainer only. Do not download, build or modify images or the `inlab-apptainer` definitions;
  ask for a `.sif` path when image mode needs testing.
- **Keep `STATUS.md` current**: what was done, what is next, open questions, new decisions (also add them to the
  decision log in `DESIGN.md`).

## Where things are

- Code (once written): `tools/nbtest/cofi_nbtest/`, run as `python -m cofi_nbtest` via the `./nbtest` wrapper
  at the repo root.
- References: `tools/nbtest/references/` (committed). Local state: `state/`, `REPORT.md`, `diffs/`,
  `.local.json` (gitignored). Staged notebooks: `examples/<dir>/<name>.nbtest.ipynb` (gitignored).
- Old runner being replaced: `tools/run_notebooks/run_notebooks.py` (+ `compare_outputs.py`).

## Note

This file is named `AGENTS.md` because the repository's `.gitignore` ignores every file named `CLAUDE.md`.
