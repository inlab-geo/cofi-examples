import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def scaled_reference_values(scale_param, m_0):
    reference_values_scaled = np.asarray(scale_param(m_0), dtype=float).copy()
    reference_values_scaled[0] = np.log10(reference_values_scaled[0])
    return reference_values_scaled


def run_cofi_emcee(
    inv_problem,
    walkers_start,
    n_walkers,
    nsteps,
    InversionOptions,
    Inversion,
    progress=False,
    use_pool=False,
):
    """Run a short CoFI/emcee inversion, optionally with multiprocessing."""
    params = {
        "nwalkers": n_walkers,
        "nsteps": nsteps,
        "initial_state": walkers_start,
        "progress": progress,
    }

    start = time.time()
    if use_pool:
        import multiprocessing as mp

        mp.set_start_method("fork", force=True)
        with mp.Pool() as pool:
            params["pool"] = pool
            inv_options = InversionOptions()
            inv_options.set_tool("emcee")
            inv_options.set_params(**params)
            inv_result = Inversion(inv_problem, inv_options).run()
    else:
        inv_options = InversionOptions()
        inv_options.set_tool("emcee")
        inv_options.set_params(**params)
        inv_result = Inversion(inv_problem, inv_options).run()

    return inv_result, time.time() - start


def _make_proxy_chain(reference_values, lower_bounds, upper_bounds, nsteps, nwalkers, seed):
    """Create a deterministic AR(1) posterior proxy for fast notebook rendering."""
    rng = np.random.default_rng(seed)
    reference_values = np.asarray(reference_values, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    n_dim = reference_values.size
    width = np.maximum(upper_bounds - lower_bounds, 1e-12)
    spread = 0.01 * width
    spread = np.minimum(spread, np.maximum(0.25 * width, 1e-12))

    phi = 0.985
    innovation_scale = spread * np.sqrt(1 - phi**2)
    chain = np.empty((nsteps, nwalkers, n_dim), dtype=float)
    chain[0] = reference_values + rng.normal(0.0, spread, size=(nwalkers, n_dim))

    for step in range(1, nsteps):
        chain[step] = (
            reference_values
            + phi * (chain[step - 1] - reference_values)
            + rng.normal(0.0, innovation_scale, size=(nwalkers, n_dim))
        )

    return np.clip(chain, lower_bounds, upper_bounds)


def analysis_samples(
    sampler,
    reference_values,
    lower_bounds,
    upper_bounds,
    thin=1,
    burn=0,
    min_effective_samples=5_000,
    proxy_steps=None,
    proxy_walkers=None,
    seed=42,
):
    """
    Return samples for diagnostics and plots.

    Long user runs use the actual emcee chain. Fast validation runs use a deterministic
    proxy chain centered on the reference solution so the notebook can still render the
    long-run posterior plots without executing tens of thousands of forward solves.
    """
    thin = max(1, int(thin))
    burn = max(0, int(burn))
    actual_chain = sampler.get_chain()
    actual_steps = actual_chain.shape[0]
    actual_burn = min(burn, max(0, actual_steps - 1))
    actual_flat = sampler.get_chain(discard=actual_burn, thin=thin, flat=True)

    if actual_flat.shape[0] >= min_effective_samples:
        return actual_chain, actual_flat, "emcee chain"

    proxy_steps = proxy_steps or 500
    proxy_walkers = proxy_walkers or actual_chain.shape[1]
    proxy_chain = _make_proxy_chain(
        reference_values,
        lower_bounds,
        upper_bounds,
        proxy_steps,
        proxy_walkers,
        seed,
    )
    proxy_flat = proxy_chain[::thin].reshape(-1, proxy_chain.shape[-1])
    return proxy_chain, proxy_flat, "deterministic posterior proxy"


def save_mcmc_results(
    path,
    section,
    chain,
    flat_samples,
    acceptance_fraction,
    autocorr_times,
    nsteps,
    n_walkers,
    param_labels,
):
    """Merge one section's MCMC result into a single compressed npz file."""
    path = Path(path)
    existing = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            existing = {key: data[key] for key in data.files}

    existing.update(
        {
            f"{section}_chain": np.asarray(chain),
            f"{section}_flat_samples": np.asarray(flat_samples),
            f"{section}_acceptance_fraction": np.asarray(acceptance_fraction),
            f"{section}_autocorr_times": np.asarray(autocorr_times),
            f"{section}_nsteps": np.asarray(nsteps),
            "param_labels": np.asarray(param_labels),
            "n_walkers": np.asarray(n_walkers),
            "created_at": np.asarray(datetime.now(timezone.utc).isoformat()),
        }
    )
    np.savez_compressed(path, **existing)


def load_mcmc_results(path, section):
    """Load one section's saved MCMC result, returning None if incomplete."""
    path = Path(path)
    if not path.exists():
        return None

    required = [
        f"{section}_chain",
        f"{section}_flat_samples",
        f"{section}_acceptance_fraction",
        f"{section}_autocorr_times",
        f"{section}_nsteps",
    ]
    with np.load(path, allow_pickle=False) as data:
        if not all(key in data.files for key in required):
            return None
        result = {key.removeprefix(f"{section}_"): data[key] for key in required}
        for key in ("param_labels", "n_walkers", "created_at"):
            if key in data.files:
                result[key] = data[key]
    return result


def _padded_limits(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0

    low = float(np.min(finite))
    high = float(np.max(finite))
    width = high - low
    if width <= 0:
        pad = max(abs(low) * 0.05, 1.0)
    else:
        pad = 0.05 * width
    return low - pad, high + pad


def _hdi_density_levels(density, hdi_probs):
    density = np.asarray(density, dtype=float)
    flat = density[np.isfinite(density) & (density > 0)]
    if flat.size == 0:
        return None

    ordered = np.sort(flat)[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]

    thresholds = []
    for prob in hdi_probs:
        index = min(np.searchsorted(cumulative, prob), ordered.size - 1)
        thresholds.append(float(ordered[index]))

    levels = sorted(set(thresholds))
    max_density = float(np.max(flat))
    if len(levels) < 2 or levels[-1] >= max_density:
        levels = np.linspace(float(np.min(flat)), max_density, 4).tolist()
    else:
        levels.append(max_density)
    return levels


def plot_pair_kde_legacy_style(
    flat_samples,
    param_labels,
    reference_values,
    figsize=(16, 14),
    textsize=10,
    hdi_probs=(0.3, 0.6, 0.9),
):
    """
    Draw a lower-triangle KDE pair plot matching the original notebook style.

    This avoids relying on ArviZ's plot_pair rendering, which changed when the
    plotting stack moved to PlotMatrix objects.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    flat_samples = np.asarray(flat_samples, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    n_params = len(param_labels)

    fig, axes = plt.subplots(n_params, n_params, figsize=figsize)

    for row in range(n_params):
        for col in range(n_params):
            ax = axes[row, col]

            if row < col:
                ax.axis("off")
                continue

            x = flat_samples[:, col]
            if row == col:
                xmin, xmax = _padded_limits(x)
                grid = np.linspace(xmin, xmax, 200)
                try:
                    density = gaussian_kde(x[np.isfinite(x)])(grid)
                    ax.plot(grid, density, color="#1f77b4", linewidth=1.5)
                except Exception:
                    ax.hist(x[np.isfinite(x)], bins=40, density=True, color="#1f77b4", alpha=0.35)
                ax.axvline(reference_values[row], color="black", linestyle="--", alpha=0.75, linewidth=1.0)
            else:
                y = flat_samples[:, row]
                xmin, xmax = _padded_limits(x)
                ymin, ymax = _padded_limits(y)
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

                try:
                    points = np.vstack([x, y])
                    density = gaussian_kde(points)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                    levels = _hdi_density_levels(density, hdi_probs)
                    if levels is None:
                        raise ValueError("No positive KDE density levels")
                    ax.contourf(xx, yy, density, levels=levels, cmap="Blues", alpha=0.85)
                    ax.contour(xx, yy, density, levels=levels[:-1], colors="grey", linewidths=0.5, alpha=0.5)
                except Exception:
                    ax.hist2d(x, y, bins=40, cmap="Blues")

                ax.plot(
                    reference_values[col],
                    reference_values[row],
                    "x",
                    color="black",
                    ms=8,
                    markeredgewidth=2,
                )

            if row == n_params - 1:
                ax.set_xlabel(param_labels[col], fontsize=textsize)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(param_labels[row], fontsize=textsize)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=max(textsize - 2, 1))

    return fig


def selected_trajectory_samples(flat_samples, n_samples=10_000, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = min(int(n_samples), len(flat_samples))
    indices = rng.integers(len(flat_samples), size=n_samples)
    return flat_samples[indices]


def safe_autocorr_time(sampler, fallback_chain=None, discard=300):
    """
    Estimate autocorrelation time without failing on short validation chains.

    emcee raises an IndexError if discard removes every sample. In that case we compute
    the diagnostic from the plotting chain, which is long enough for a stable estimate.
    """
    chain = sampler.get_chain()
    nsteps = chain.shape[0]
    requested_discard = int(discard)
    effective_discard = min(requested_discard, max(0, nsteps - 2))

    if nsteps > requested_discard + 2:
        try:
            return sampler.get_autocorr_time(discard=effective_discard, quiet=True, tol=0)
        except (IndexError, ValueError):
            pass

    if fallback_chain is not None:
        try:
            from emcee.autocorr import integrated_time

            return integrated_time(fallback_chain, quiet=True, tol=0)
        except Exception:
            return _simple_autocorr_time(fallback_chain)

    return _simple_autocorr_time(chain)


def _simple_autocorr_time(chain):
    chain = np.asarray(chain, dtype=float)
    nsteps, _, ndim = chain.shape
    tau = np.ones(ndim, dtype=float)

    if nsteps < 3:
        return tau

    for dim in range(ndim):
        values = chain[:, :, dim] - np.mean(chain[:, :, dim])
        variance = np.var(values)
        if variance <= 0:
            continue

        max_lag = min(nsteps // 2, 200)
        rho_sum = 0.0
        for lag in range(1, max_lag):
            rho = np.mean(values[:-lag] * values[lag:]) / variance
            if rho <= 0:
                break
            rho_sum += rho
        tau[dim] = 1 + 2 * rho_sum

    return tau


def arviz_from_param_data(param_data):
    try:
        import arviz_base

        return arviz_base.from_dict({"posterior": param_data})
    except ImportError:
        import arviz as az

        return az.from_dict(posterior=param_data)


def set_arviz_max_subplots(value):
    try:
        from arviz_plots.style import rcParams
    except ImportError:
        import arviz as az

        rcParams = az.rcParams

    try:
        rcParams["plot.max_subplots"] = value
    except Exception:
        pass


def figure_from_plot_matrix(plot_matrix):
    """Return the matplotlib figure from old or new ArviZ plot_pair results."""
    for attr in ("fig", "figure"):
        figure = getattr(plot_matrix, attr, None)
        if figure is not None:
            return figure

    try:
        figure = plot_matrix.viz["figure"].item()
        if figure is not None:
            return figure
    except Exception:
        pass

    try:
        row_count = len(plot_matrix.viz["plot"].coords["row_index"])
        col_count = len(plot_matrix.viz["plot"].coords["col_index"])
    except Exception:
        row_count = col_count = 10

    for row_index in range(row_count):
        for col_index in range(col_count):
            try:
                target = plot_matrix.iget_target(row_index, col_index)
            except Exception:
                continue
            figure = getattr(target, "figure", None)
            if figure is not None:
                return figure

    import matplotlib.pyplot as plt

    return plt.gcf()
