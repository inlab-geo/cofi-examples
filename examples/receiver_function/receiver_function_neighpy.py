"""Receiver function inversion with the Neighbourhood Algorithm via CoFI

This script demonstrates the Neighbourhood Algorithm (NA) applied to receiver
function inversion through CoFI's neighpy wrapper. The two stages of the NA
are run separately:

- Stage I (neighpyI): Direct search that minimises a misfit function
- Stage II (neighpyII): Appraisal that resamples the Stage I ensemble according
  to a user-supplied log posterior probability density

The forward model uses pyrf96 (Shibutani et al. 1996) to compute synthetic
receiver functions from a 4-layer Earth model parameterised as Voronoi nuclei
[depth_km, Vs_km/s, Vp/Vs]. We invert for 8 parameters (4 depths + 4 Vs),
with Vp/Vs ratios held fixed.
"""

############# 0. Import modules #######################################################

import math
import numpy as np
import matplotlib.pyplot as plt

from cofi import BaseProblem, InversionOptions, Inversion
import pyrf96

np.random.seed(42)

save_plot = True
show_plot = False
show_summary = True

_problem_name = "receiver_function"
_solver_name = "neighpy"
_file_prefix = f"{_problem_name}_{_solver_name}"


def plot_model_staircase(model, ax, max_depth=60.0, **kwargs):
    """Plot a pyrf96 Voronoi nuclei model as a Vs-depth staircase."""
    order = np.argsort(model[:, 0])
    depths = model[order, 0]
    vs = model[order, 1]
    interfaces = 0.5 * (depths[:-1] + depths[1:])
    plot_depths = np.concatenate([[0.0], np.repeat(interfaces, 2), [max_depth]])
    plot_vs = np.repeat(vs, 2)
    ax.plot(plot_vs, plot_depths, **kwargs)


def main(output_dir="."):
    _figs_prefix = f"{output_dir}/{_file_prefix}"

    ######### 1. Define the problem ###################################################

    good_model = np.array([
        [ 1.0, 3.0, 1.7],
        [ 8.0, 3.2, 2.0],
        [20.0, 4.0, 1.7],
        [45.0, 4.2, 1.7],
    ])

    vpvs = good_model[:, 2]

    def get_inversion_parameters(fullmodel):
        return fullmodel[:, :2].flatten()

    def get_model_parameters(invmodel):
        return np.append(invmodel.reshape(len(vpvs), -1), vpvs[:, None], axis=1)

    # Generate synthetic data
    t, rfunc = pyrf96.rfcalc(good_model)
    t2, rfunc_noise = pyrf96.rfcalc(good_model, sn=0.5, seed=12345678)
    observed_data = rfunc_noise
    Cdinv = pyrf96.InvDataCov(2.5, 0.01, len(rfunc))

    # Misfit function (used by Stage I)
    def misfit_dv(imodel):
        model = get_model_parameters(imodel)
        t_pred, predicted_data = pyrf96.rfcalc(model)
        res = observed_data - predicted_data
        misfit_val = np.dot(res, np.transpose(np.dot(Cdinv, res))) / 2.0
        if math.isnan(misfit_val):
            return float("inf")
        return misfit_val

    # Log posterior function (used by Stage II)
    def log_posterior_dv(imodel):
        """Log posterior = -misfit (uniform prior, Gaussian likelihood)."""
        return -misfit_dv(imodel)

    # CoFI BaseProblem
    inv_problem = BaseProblem()
    inv_problem.name = "Receiver Function - Depth + Velocity"
    inv_problem.set_objective(misfit_dv)
    if show_summary:
        inv_problem.summary()

    ######### 2. Stage I: Direct Search ###############################################

    bounds_dv = [(0.0, 60.0), (2.0, 4.5)] * 4

    inv_options_ds = InversionOptions()
    inv_options_ds.set_tool("neighpyI")
    inv_options_ds.set_params(
        bounds=bounds_dv,
        n_initial_samples=5000,
        n_samples_per_iteration=200,
        n_cells_to_resample=50,
        n_iterations=100,
    )
    if show_summary:
        inv_options_ds.summary()

    inv_ds = Inversion(inv_problem, inv_options_ds)
    inv_result_ds = inv_ds.run()
    if show_summary:
        inv_result_ds.summary()

    best = inv_result_ds.model
    ds_samples = inv_result_ds.samples
    ds_objectives = inv_result_ds.objectives

    ######### 3. Stage II: Appraisal ##################################################

    # Compute log-PPD: log_posterior = -misfit (Gaussian likelihood, uniform prior)
    log_ppd = -ds_objectives

    inv_options_app = InversionOptions()
    inv_options_app.set_tool("neighpyII")
    inv_options_app.set_params(
        bounds=bounds_dv,
        initial_ensemble=ds_samples,
        log_ppd=log_ppd,
        n_resample=50000,
        n_walkers=10,
    )
    if show_summary:
        inv_options_app.summary()

    inv_app = Inversion(inv_problem, inv_options_app)
    inv_result_app = inv_app.run()
    if show_summary:
        inv_result_app.summary()

    appraisal_samples = inv_result_app.new_samples
    appraisal_mean = appraisal_samples.mean(axis=0)
    true_dv = get_inversion_parameters(good_model)

    ######### 4. Plot results #########################################################

    if save_plot or show_plot:
        # Plot 1: Convergence
        best_i = np.argmin(ds_objectives)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ds_objectives, marker=".", linestyle="", markersize=2, color="black")
        ax.scatter(best_i, ds_objectives[best_i], c="g", s=30, zorder=10,
                   label="Best model")
        ax.axvline(5000, c="grey", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Sample number")
        ax.set_ylabel("Misfit")
        ax.set_title("NA-I convergence (depth + velocity)")
        ax.legend()
        if save_plot:
            fig.savefig(f"{_figs_prefix}_convergence")

        # Plot 2: Model ensemble
        fig2, ax2 = plt.subplots(figsize=(5, 8))
        for i in range(0, len(appraisal_samples), 10):
            m = get_model_parameters(appraisal_samples[i])
            plot_model_staircase(m, ax2, color="k", alpha=0.01, lw=0.5)
        plot_model_staircase(good_model, ax2, color="b", lw=2, label="True model")
        plot_model_staircase(get_model_parameters(best), ax2, color="g", lw=2,
                             ls="--", label="Best model (NA-I)")
        plot_model_staircase(get_model_parameters(appraisal_mean), ax2, color="r",
                             lw=2, ls="--", label="Mean model (NA-II)")
        ax2.set_xlim(2.5, 5.0)
        ax2.set_ylim(60, 0)
        ax2.set_xlabel("Vs (km/s)")
        ax2.set_ylabel("Depth (km)")
        ax2.set_title("NA-II appraisal ensemble")
        ax2.legend(loc="lower left")
        if save_plot:
            fig2.savefig(f"{_figs_prefix}_models")

        # Plot 3: Predicted data
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        for i in range(0, len(appraisal_samples), 15):
            m = get_model_parameters(appraisal_samples[i])
            try:
                t_pred, rf_pred = pyrf96.rfcalc(m)
                ax3.plot(t_pred, rf_pred, color="k", alpha=0.01, lw=0.5)
            except Exception:
                pass
        ax3.plot(t2, observed_data, color="k", lw=1.5, label="Observed", zorder=10)
        t_best, rf_best = pyrf96.rfcalc(get_model_parameters(best))
        ax3.plot(t_best, rf_best, color="g", ls="--", lw=1.5,
                 label="Best model (NA-I)", zorder=11)
        t_mean, rf_mean = pyrf96.rfcalc(get_model_parameters(appraisal_mean))
        ax3.plot(t_mean, rf_mean, color="r", ls="--", lw=1.5,
                 label="Mean model (NA-II)", zorder=11)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Amplitude")
        ax3.set_title("Predicted receiver functions from NA-II ensemble")
        ax3.legend()
        if save_plot:
            fig3.savefig(f"{_figs_prefix}_data")

        # Plot 4: 8x8 corner plot
        n_params = 8
        var_names = ["d1", "Vs1", "d2", "Vs2", "d3", "Vs3", "d4", "Vs4"]
        fig4, axes = plt.subplots(n_params, n_params, figsize=(12, 12),
                                  tight_layout=True,
                                  gridspec_kw={"hspace": 0, "wspace": 0})
        for i in range(n_params):
            for j in range(n_params):
                axes[i, j].set_xlim(bounds_dv[j])
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                if i == j:
                    axes[i, j].hist(appraisal_samples[::10, j], bins=100,
                                    histtype="step", color="k")
                    axes[i, j].axvline(best[j], c="g", ls="--")
                    axes[i, j].axvline(true_dv[j], c="b", ls="--")
                    axes[i, j].axvline(appraisal_mean[j], c="r", ls="--")
                elif j < i:
                    axes[i, j].scatter(appraisal_samples[::10, j],
                                       appraisal_samples[::10, i],
                                       s=1, c="k", alpha=0.01)
                    axes[i, j].scatter(best[j], best[i], c="g", s=10, zorder=10)
                    axes[i, j].scatter(true_dv[j], true_dv[i], c="b", s=10,
                                       zorder=10)
                    axes[i, j].scatter(appraisal_mean[j], appraisal_mean[i],
                                       c="r", s=10, zorder=10)
                    axes[i, j].set_ylim(bounds_dv[i])
                else:
                    axes[i, j].axis("off")
            if j <= i:
                axes[-1, j].set_xlabel(var_names[j])
            if i > 0:
                axes[i, 0].set_ylabel(var_names[i])

        fig4.suptitle("Posterior corner plot (depth + velocity)")
        if save_plot:
            fig4.savefig(f"{_figs_prefix}_corner")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Receiver function inversion with the Neighbourhood Algorithm via CoFI"
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
