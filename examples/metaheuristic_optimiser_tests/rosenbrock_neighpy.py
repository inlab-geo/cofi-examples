"""Rosenbrock function optimised with the Neighbourhood Algorithm via CoFI

This script demonstrates the Neighbourhood Algorithm (NA) applied to the
Rosenbrock test function through CoFI's neighpy wrapper. The NA performs a
derivative-free direct search followed by an appraisal phase that resamples
the parameter space without additional objective evaluations.

The Rosenbrock function is:
    f(x, y) = (a - x)^2 + b * (y - x^2)^2
with a=1, b=100 and global minimum at (1, 1).

We use the log10-scaled version as the objective to avoid very large values.
"""

############# 0. Import modules #######################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi, voronoi_plot_2d

from cofi import BaseProblem, InversionOptions, Inversion

np.random.seed(42)

save_plot = True
show_plot = False
show_summary = True

_problem_name = "rosenbrock"
_solver_name = "neighpy"
_file_prefix = f"{_problem_name}_{_solver_name}"


def main(output_dir="."):
    _figs_prefix = f"{output_dir}/{_file_prefix}"

    ######### 1. Define the problem ###################################################

    def rosenbrock(params, a=1, b=100):
        return np.log10((a - params[0])**2 + b * (params[1] - params[0]**2)**2)

    inv_problem = BaseProblem()
    inv_problem.name = "Rosenbrock Function"
    inv_problem.set_objective(rosenbrock)
    if show_summary:
        inv_problem.summary()

    ######### 2. Define the inversion options #########################################

    bounds = [(-2, 2), (-1, 3)]
    n_initial_samples = 100
    n_samples_per_iteration = 70
    n_cells_to_resample = 10
    n_iterations = 20
    n_resample = 50000
    n_walkers = 10

    inv_options = InversionOptions()
    inv_options.set_tool("neighpy")
    inv_options.set_params(
        bounds=bounds,
        n_initial_samples=n_initial_samples,
        n_samples_per_iteration=n_samples_per_iteration,
        n_cells_to_resample=n_cells_to_resample,
        n_iterations=n_iterations,
        n_resample=n_resample,
        n_walkers=n_walkers,
    )
    if show_summary:
        inv_options.summary()

    ######### 3. Start an inversion ###################################################

    inv = Inversion(inv_problem, inv_options)
    inv_result = inv.run()
    if show_summary:
        inv_result.summary()

    best = inv_result.model
    ds_samples = inv_result.direct_search_samples
    appraisal_samples = inv_result.appraisal_samples

    ######### 4. Plot results #########################################################

    if save_plot or show_plot:
        # Contour data
        X, Y = np.meshgrid(np.linspace(-2, 2, 500), np.linspace(-1, 3, 500))
        Z = np.log10((1 - X)**2 + 100 * (Y - X**2)**2)

        # Voronoi + contour plot
        fig = voronoi_plot_2d(
            Voronoi(ds_samples), show_vertices=False, line_width=0.5, line_colors="w"
        )
        ax = fig.gca()
        im = ax.imshow(Z, origin="lower", extent=(-2, 2, -1, 3), aspect="auto")
        fig.colorbar(im)
        _truth = ax.scatter(1, 1, c="r", marker="x", s=100, zorder=10, label="True minimum")
        _best = ax.scatter(*best, c="k", marker="+", s=100, zorder=10, label="Best sample (NA-I)")
        _resample = ax.scatter(
            *appraisal_samples.T, s=0.5, c="grey", zorder=0, label="Resampled points (NA-II)"
        )
        _voronoi = Line2D(
            [0], [0], marker="o", label="Voronoi samples (NA-I)", markersize=5, linewidth=0
        )
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(
            handles=[_truth, _best, _voronoi, _resample], framealpha=1, edgecolor="black"
        )
        ax.set_title("NA on Rosenbrock function")
        if save_plot:
            fig.savefig(f"{_figs_prefix}_voronoi")

        # Appraisal scatter plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im2 = ax2.imshow(Z, origin="lower", extent=(-2, 2, -1, 3), aspect="auto")
        fig2.colorbar(im2)
        ax2.scatter(*appraisal_samples.T, s=0.5, c="grey", label="Resampled points (NA-II)")
        ax2.scatter(1, 1, c="r", marker="x", s=100, zorder=10, label="True minimum")
        ax2.scatter(*best, c="k", marker="+", s=100, zorder=10, label="Best sample (NA-I)")
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-1, 3)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.legend(framealpha=1, edgecolor="black")
        ax2.set_title("NA-II Appraisal samples on Rosenbrock function")
        if save_plot:
            fig2.savefig(f"{_figs_prefix}_appraisal")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Rosenbrock function optimised with the Neighbourhood Algorithm via CoFI"
    )
    parser.add_argument("--output-dir", "-o", type=str, help="output folder for figures")
    parser.add_argument("--show-plot", dest="show_plot", action="store_true", default=None)
    parser.add_argument("--no-show-plot", dest="show_plot", action="store_false", default=None)
    parser.add_argument("--save-plot", dest="save_plot", action="store_true", default=None)
    parser.add_argument("--no-save-plot", dest="save_plot", action="store_false", default=None)
    parser.add_argument("--show-summary", dest="show_summary", action="store_true", default=None)
    parser.add_argument("--no-show-summary", dest="show_summary", action="store_false", default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or "."
    if output_dir.endswith("/"):
        output_dir = output_dir[:-1]
    show_plot = show_plot if args.show_plot is None else args.show_plot
    save_plot = save_plot if args.save_plot is None else args.save_plot
    show_summary = show_summary if args.show_summary is None else args.show_summary

    main(output_dir)
