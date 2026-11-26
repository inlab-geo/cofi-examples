# Using nbtest

`nbtest` is the developing notebook runner and validation tool for this repository. The current implementation
runs notebooks locally into disposable staged copies, protects those copies from accidental replacement, and
records the latest run. Numerical and figure validation, reference blessing, publishing, image mode and full
environment provenance are later slices and are not available yet.

The design and development record live in [`DESIGN.md`](DESIGN.md) and [`STATUS.md`](STATUS.md).

## Run notebooks

Run from the repository root with the Python environment whose packages and `cofi` installation you want to
exercise:

```console
./nbtest run examples/nonlinear_curve_fitting/nonlinear_curve_fitting.ipynb
./nbtest run examples/linear_regression
./nbtest run @smoke
./nbtest run all
```

Targets can be `all`, one or more notebook paths, example folders, or `@name` for
`tools/nbtest/name.txt`. Notebooks listed as prerequisites in a folder's `dependencies.txt` are inserted before
the requested target and labelled `prerequisite` in the report. Use `--timeout SECONDS` to change the default
three-hour per-notebook limit.

The original notebook is never changed. Each executed notebook is written beside it as
`<name>.nbtest.ipynb`, with fresh outputs and `metadata.nbtest` describing the attempt. A failed notebook still
gets a staged copy when the kernel stopped cleanly; an interrupted notebook does not.

## Results and file locations

All generated files are ignored by Git:

| File | Purpose |
| --- | --- |
| `examples/<folder>/<name>.nbtest.ipynb` | Executed copy to inspect in Jupyter. |
| `tools/nbtest/state/state.json` | Machine-readable state: trusted staged hashes, supporting details, and the latest run. |
| `tools/nbtest/REPORT.md` | Human-readable summary of the latest run only. |

`state.json` and `REPORT.md` are written atomically before execution, after every completed notebook, and when
a run completes, is interrupted, or aborts unexpectedly. The latest run keeps the complete ordered plan, so an
interruption shows the active notebook as `interrupted` and notebooks not reached as `not_run`.

Rebuild the report from saved state without running a notebook or importing notebook-execution packages:

```console
./nbtest report
```

The report includes targets, UTC dates, exit code, role, status, duration, concise failure details and files
created, modified or deleted by each notebook. A change to a Git-tracked file is highlighted as a warning.
Git-ignored files such as `__pycache__/`, the expected staged notebook, and completed atomic temporary files are
not reported as notebook side effects.

## Protecting staged work

Before starting a kernel, `run` checks every selected staged notebook against its saved SHA-256. It refuses to
replace a staged file that has been manually edited or was created before state tracking. Review or copy out any
work you need, then explicitly replace it with:

```console
./nbtest run --force <targets>
```

`--force` is reported in the terminal but is not saved as state. It does not suppress notebook failures.

If `state.json` is corrupt or uses an unsupported schema, execution stops before a kernel starts. The recovery
path is to remove `tools/nbtest/state/` and rerun with `--force` if staged notebooks already exist. Two concurrent
`nbtest run` processes are unsupported: atomic writes prevent partial files, but the last state writer wins and
the next hash preflight safely rejects any mismatch.

## Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | The command completed with no failed or timed-out notebooks. Skipped optional packages are not failures. |
| `1` | At least one notebook failed or timed out. |
| `2` | Usage, target, state, dependency or unexpected setup error. |
| `130` | Interrupted with Ctrl-C. |

The exit code describes execution only. Comparing staged results with numerical and figure references begins in
the validation slice described in `DESIGN.md`.
