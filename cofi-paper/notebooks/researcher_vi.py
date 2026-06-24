import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    # Standard library
    import h5py

    # Third-party numerical libraries
    import numpy as np
    import scipy

    # Visualisation
    import matplotlib.pyplot as plt
    import cmcrameri.cm as scm

    # Cartopy for map projections
    import cartopy.crs as ccrs

    # Inversion framework
    import cofi
    from cofi import BaseProblem, InversionOptions, Inversion

    # Notebook interface
    import marimo as mo

    return (
        BaseProblem,
        Inversion,
        InversionOptions,
        ccrs,
        cofi,
        h5py,
        mo,
        np,
        plt,
        scipy,
        scm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gaussian Variational Inference for Surface Wave Tomography

    ## Introduction

    This notebook applies **Gaussian Variational Inference** (VI) to ambient noise
    surface wave tomography across Australia, using CoFI's `cofi.gaussian_vi` solver.

    The deterministic inversions in the companion `researcher.py` notebook produce a
    single "best" model but do not quantify **uncertainty**. Gaussian VI provides a
    scalable approximation to the posterior distribution by fitting a multivariate
    Gaussian, yielding both a posterior mean and a full covariance (uncertainty) estimate.

    ### Method

    CoFI's `cofi.gaussian_vi` solver implements natural-gradient Gaussian VI in three phases:

    1. **Phase 1 — MAP initialisation**: Gauss-Newton iteration finds the maximum a posteriori
       estimate, providing a good starting point.
    2. **Phase 2 — Gaussian VI**: Natural-gradient updates on the mean and precision of a
       Gaussian approximation to the posterior, tracked via the Evidence Lower Bound (ELBO).
    3. **Phase 3 — Optional normalising flow**: A sinh-arcsinh transformation can capture
       non-Gaussian features (skewness, heavy tails) in the posterior.

    All matrices remain sparse throughout (the solver uses CHOLMOD for sparse Cholesky
    factorisation), making this feasible for the ~19,000-parameter grid.

    ### Prior and data covariance

    The prior precision $\mathbf{Q} = \mathbf{R}^\top\mathbf{R}$ is built from an SPDE
    Matérn operator with physically meaningful parameters:

    - **$\sigma$** (prior std dev) encodes how much slowness can deviate from the background
    - **$\ell$** (correlation length) controls spatial smoothness

    The data-vs-prior balance is controlled through the **data covariance** $\mathbf{C}_d$,
    not by scaling Q. This keeps the prior physically interpretable: $\sigma$ means what it
    says, and tightening $\mathbf{C}_d$ (reducing assumed data noise) increases the data's
    relative weight.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data

    The dataset comprises 5-second period Rayleigh wave phase velocity measurements from
    ambient noise cross-correlations between ~1,100 seismic stations across Australia
    (Magrini et al. 2023). After deduplication, this provides ~15,000 inter-station
    travel time observations.
    """)
    return


@app.cell
def _(h5py, np):
    # Load data from HDF5 file
    from pathlib import Path as _Path
    with h5py.File(_Path(__file__).resolve().parent.parent / "data" / "sw_tomography.h5", "r") as f:
        observations = f["observations"][:]      # (15661,) slowness in s/m
        stations = f["stations"][:]              # (1122, 2) [lat, lon]
        station_pairs = f["station_pairs"][:]    # (15661, 2) [sta1_idx, sta2_idx]

    print(f"Using all {len(station_pairs):,} rays")

    # Remove duplicate station pairs (average observations for duplicates)
    unique_pairs, inverse_indices = np.unique(station_pairs, axis=0, return_inverse=True)
    if len(unique_pairs) < len(station_pairs):
        # Average observations for duplicate pairs
        unique_obs = np.zeros(len(unique_pairs))
        counts = np.zeros(len(unique_pairs))
        for obs_idx, unique_idx in enumerate(inverse_indices):
            unique_obs[unique_idx] += observations[obs_idx]
            counts[unique_idx] += 1
        unique_obs /= counts
        print(f"  Removed {len(station_pairs) - len(unique_pairs)} duplicate pairs (averaged observations)")
        station_pairs = unique_pairs
        observations = unique_obs
    return observations, station_pairs, stations


@app.cell
def _():
    # Define grid parameters
    grid_resolution = 0.3  # degrees
    latmin, latmax = -46.2, -8.1
    lonmin, lonmax = 110.9, 156.2
    return grid_resolution, latmax, latmin, lonmax, lonmin


@app.cell
def _(np, observations, station_pairs, stations):
    # Convert observed slowness to travel times using great circle distances
    # observations is slowness in s/m; convert to travel times in seconds
    R_earth = 6371000.0

    def haversine_distance(lat1, lon1, lat2, lon2):
        """Compute great circle distance in meters."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R_earth * np.arcsin(np.sqrt(a))

    distances = np.array([
        haversine_distance(
            stations[s1, 0], stations[s1, 1],
            stations[s2, 0], stations[s2, 1]
        )
        for s1, s2 in station_pairs
    ])

    d_obs_tt = observations * distances
    print(f"Observed travel times computed from real data:")
    print(f"  Distance range: [{distances.min()/1000:.1f}, {distances.max()/1000:.1f}] km")
    print(f"  Travel time range: [{d_obs_tt.min():.1f}, {d_obs_tt.max():.1f}] s")
    return (d_obs_tt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forward model: pyfm2d ray tracing

    Travel times are computed using **pyfm2d**, which solves the eikonal equation via a
    fast marching method. Unlike the linear (great circle) approximation, pyfm2d accounts
    for ray bending through the velocity structure. The Jacobian (Fréchet derivative) is
    computed alongside the travel times for use in the Gauss-Newton updates within VI.
    """)
    return


@app.cell
def _():
    import pyfm2d

    return


@app.cell
def _(
    grid_resolution,
    latmax,
    latmin,
    lonmax,
    lonmin,
    np,
    station_pairs,
    stations,
):
    # Grid extent for pyfm2d [lonmin, lonmax, latmin, latmax]
    # Matches the seislib grid exactly — all stations are at least 2.7° inside
    # these bounds, so no additional buffer is needed for pyfm2d.
    fm2d_extent = [lonmin, lonmax, latmin, latmax]

    # Compute grid shape from extent and resolution
    # pyfm2d expects velocity as (nx, ny) = (n_lon, n_lat)
    n_lon = int(np.round((fm2d_extent[1] - fm2d_extent[0]) / grid_resolution)) + 1
    n_lat = int(np.round((fm2d_extent[3] - fm2d_extent[2]) / grid_resolution)) + 1
    fm2d_grid_shape = (n_lon, n_lat)

    # Note: the source-receiver associations below cover the same geometry as the
    # linear inversion, but seislib handles that internally (inside compile_coefficients)
    # and does not expose it in a form pyfm2d can use. pyfm2d requires an explicit
    # associations matrix, local index mappings, and a reorder array, so the
    # geometry must be reconstructed here from the same raw station_pairs data.

    # Get unique sources and receivers from (possibly subsetted) station_pairs
    unique_src_idx = np.unique(station_pairs[:, 0])
    unique_rec_idx = np.unique(station_pairs[:, 1])

    # Create index mappings
    src_map = {sta_idx: local_idx for local_idx, sta_idx in enumerate(unique_src_idx)}
    rec_map = {sta_idx: local_idx for local_idx, sta_idx in enumerate(unique_rec_idx)}

    # Build associations matrix
    n_src = len(unique_src_idx)
    n_rec = len(unique_rec_idx)
    fm2d_associations = np.zeros((n_rec, n_src), dtype=np.int32)
    for src_idx, rec_idx in station_pairs:
        fm2d_associations[rec_map[rec_idx], src_map[src_idx]] = 1

    # Sources and receivers in [lon, lat] format for pyfm2d
    fm2d_sources = stations[unique_src_idx][:, ::-1]
    fm2d_receivers = stations[unique_rec_idx][:, ::-1]

    # Build mapping from pyfm2d output order to station_pairs order
    # pyfm2d with multithreading outputs: for each source, receivers with associations==1
    # Map this back to station_pairs order
    pyfm2d_to_pairs = {}  # (src_local, rec_local) -> index in station_pairs
    for pair_idx, (src_idx, rec_idx) in enumerate(station_pairs):
        src_local = src_map[src_idx]
        rec_local = rec_map[rec_idx]
        pyfm2d_to_pairs[(src_local, rec_local)] = pair_idx

    # Build reorder indices: pyfm2d output order -> station_pairs order
    # pyfm2d outputs in source-major order: for each source, list receivers
    fm2d_reorder = []
    for src_local in range(n_src):
        for rec_local in range(n_rec):
            if fm2d_associations[rec_local, src_local] == 1:
                fm2d_reorder.append(pyfm2d_to_pairs[(src_local, rec_local)])
    fm2d_reorder = np.array(fm2d_reorder)

    print(f"pyfm2d configuration:")
    print(f"  Grid shape: {fm2d_grid_shape} ({n_lat} lat x {n_lon} lon = {n_lat * n_lon:,} cells)")
    print(f"  Sources: {n_src}, Receivers: {n_rec}, Pairs: {len(station_pairs):,}")
    return (
        fm2d_associations,
        fm2d_extent,
        fm2d_grid_shape,
        fm2d_receivers,
        fm2d_reorder,
        fm2d_sources,
    )


@app.cell(hide_code=True)
def _():
    # Write worker module for parallel pyfm2d execution.
    # ProcessPoolExecutor requires functions to be importable from a module,
    # so this file is generated in a scratch directory.
    import sys as _sys
    from pathlib import Path
    from textwrap import dedent

    _worker_code = dedent("""
        \"\"\"Worker function for parallel pyfm2d ray tracing.

        This module is auto-generated by the notebook because ProcessPoolExecutor
        requires picklable functions defined at module level.
        \"\"\"

        import numpy as np


        def compute_single_source(args):
            \"\"\"Compute travel times for a single source using pyfm2d.

            Parameters
            ----------
            args : tuple
                (src_idx, velocity, receivers, source, extent, return_jacobian)

            Returns
            -------
            tuple
                (src_idx, ttimes, frechet_data) where frechet_data is tuple for pickling
            \"\"\"
            src_idx, velocity, receivers, source, extent, return_jacobian = args

            import pyfm2d
            from pyfm2d import WaveTrackerOptions

            options = WaveTrackerOptions(
                times=True,
                frechet=return_jacobian,
                paths=return_jacobian,
                cartesian=False,
                quiet=True,
            )

            result = pyfm2d.calc_wavefronts(
                velocity,
                receivers,
                source,
                extent=extent,
                options=options,
                nthreads=1,
            )

            if return_jacobian:
                frechet = result.frechet.tocsr()
                frechet_data = (frechet.data.copy(), frechet.indices.copy(),
                                frechet.indptr.copy(), frechet.shape)
                return src_idx, result.ttimes.copy(), frechet_data
            return src_idx, result.ttimes.copy(), None
    """).lstrip()

    # Create temp directory and write worker module there
    import tempfile

    scratch_dir = Path(tempfile.mkdtemp(prefix="cofi_beans_"))
    pyfm2d_worker_path = scratch_dir / "pyfm2d_worker.py"
    pyfm2d_worker_path.write_text(_worker_code)

    # Add temp directory to sys.path for import
    scratch_str = str(scratch_dir)
    if scratch_str not in _sys.path:
        _sys.path.insert(0, scratch_str)

    pyfm2d_worker_ready = True
    return pyfm2d_worker_ready, scratch_dir


@app.cell(hide_code=True)
def _(
    fm2d_associations,
    fm2d_extent,
    fm2d_grid_shape,
    fm2d_receivers,
    fm2d_reorder,
    fm2d_sources,
    np,
    pyfm2d_worker_ready,
    scipy,
):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm.auto import tqdm
    from pyfm2d_worker import compute_single_source
    _uses = pyfm2d_worker_ready  # Ensure worker file is written first
    import os

    # Determine number of parallel workers (use 25% of available CPUs to limit thermal load)
    n_workers = max(1, (os.cpu_count() or 1) // 4)

    def forward_pyfm2d(slowness_model, return_jacobian=False, progress=True):
        """Compute travel times (and optionally Jacobian) using pyfm2d.

        Uses ProcessPoolExecutor for parallel execution over sources with progress reporting.
        Process-based parallelism is required because pyfm2d's C/Fortran code is not thread-safe.
        Workers are set to 25% of os.cpu_count() to limit thermal load.

        Parameters
        ----------
        slowness_model : array-like
            Slowness model as a 1D array (n_cells,) in s/km.
        return_jacobian : bool
            If True, return (travel_times, jacobian).
        progress : bool
            If True, show progress bar.

        Returns
        -------
        travel_times : ndarray
            Predicted travel times for each source-receiver pair, in station_pairs order.
        jacobian : sparse matrix (optional)
            Frechet derivative matrix (n_obs, n_cells).
        """
        # Convert slowness (s/km) to velocity (km/s) and reshape to grid
        velocity = 1.0 / slowness_model.reshape(fm2d_grid_shape)
        n_sources = fm2d_sources.shape[0]

        # Parallel execution over sources with progress bar
        # Using ProcessPoolExecutor because pyfm2d C/Fortran code is not thread-safe
        all_ttimes = {}
        all_frechet = {} if return_jacobian else None

        # Build task arguments as tuples for the worker function
        tasks = []
        for src_idx in range(n_sources):
            rec_mask = fm2d_associations[:, src_idx] == 1
            if not np.any(rec_mask):
                continue
            source = fm2d_sources[src_idx:src_idx+1]
            receivers = fm2d_receivers[rec_mask]
            tasks.append((src_idx, velocity, receivers, source, fm2d_extent, return_jacobian))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(compute_single_source, task): task[0] for task in tasks}

            iterator = as_completed(futures)
            if progress:
                iterator = tqdm(iterator, total=len(futures), desc="Ray tracing", unit="src")

            for future in iterator:
                src_idx, ttimes, frechet_data = future.result()
                if ttimes is not None:
                    all_ttimes[src_idx] = ttimes
                    if return_jacobian and frechet_data is not None:
                        all_frechet[src_idx] = frechet_data

        # Combine results in source order and reorder to station_pairs order
        ttimes_list = []
        frechet_list = [] if return_jacobian else None

        for src_idx in range(n_sources):
            if src_idx in all_ttimes:
                ttimes_list.append(all_ttimes[src_idx])
                if return_jacobian:
                    # Reconstruct sparse matrix from tuple (data, indices, indptr, shape)
                    data, indices, indptr, shape = all_frechet[src_idx]
                    frechet_mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
                    frechet_list.append(frechet_mat)

        ttimes_combined = np.concatenate(ttimes_list)
        ttimes_out = np.empty(len(fm2d_reorder))
        ttimes_out[fm2d_reorder] = ttimes_combined

        if return_jacobian:
            frechet_combined = scipy.sparse.vstack(frechet_list)
            frechet_out = frechet_combined.tocsr()[np.argsort(fm2d_reorder), :]
            return ttimes_out, frechet_out
        return ttimes_out

    return (forward_pyfm2d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SPDE Matern regularisation (prior)

    The prior precision is built from the SPDE Matern v=1 factor:

    $$\mathbf{R} = \tau \sqrt{h_x h_y}(\kappa^2 \mathbf{I} - \mathbf{L}_h), \quad \kappa = \sqrt{2}/\ell, \quad \tau = \frac{1}{2\sqrt{\pi}\,\kappa\,\sigma}$$

    The prior precision is $\mathbf{Q} = \mathbf{R}^\top\mathbf{R}$, with marginal standard
    deviation exactly $\sigma$ by construction. This $\sigma$ should encode a genuine
    physical belief about plausible velocity perturbations.

    At a background velocity of 3 km/s, the observed velocity range [2.1, 3.6] km/s
    corresponds to slowness deviations up to ~0.14 s/km. Setting $\sigma = 0.07$ s/km
    places this range within ~2$\sigma$, consistent with plausible Rayleigh wave phase
    velocity variations across continental Australia at 5 s period.

    The **data-vs-prior balance** is controlled through $\mathbf{C}_d$ (data noise),
    not by scaling Q. Tighter $\mathbf{C}_d$ increases data weight; wider $\sigma$
    relaxes the prior.
    """)
    return


@app.cell
def _(grid_resolution, np):
    # SPDE Matern v=1 parameters
    ell = np.sqrt(2) * 5 * grid_resolution  # Matern length scale in degrees (practical range rho = 2*ell)
    # Prior std dev of slowness perturbations (s/km).
    # Sets max observed deviation (~0.11 s/km) at ~2.75 sigma, so the prior
    # actively shapes the posterior rather than being vacuous.
    sigma = 0.04
    return ell, sigma


@app.cell(hide_code=True)
def _(ell, np, plt):
    from scipy.special import kv as _kv
    _kappa = np.sqrt(2.0) / ell
    _rho = 2.0 * ell
    _r_deg = np.linspace(1e-6, 2.0 * _rho, 400)
    _x = _kappa * _r_deg
    _corr = _x * _kv(1, _x)
    _r_km = _r_deg * 111.0
    _ell_km = ell * 111.0
    _rho_km = _rho * 111.0

    _fig, _ax = plt.subplots(figsize=(5, 2.8))
    _ax.plot(_r_km, _corr, 'k-', lw=1.5)
    _ax.axvline(_ell_km, color='C1', ls='--', lw=1,
                label=f'$\\ell$ = {ell:.2f}' + f'\u00b0 \u2248 {_ell_km:.0f} km  (corr \u2248 0.44)')
    _ax.axvline(_rho_km, color='C0', ls='--', lw=1,
                label=f'$\\rho = 2\\ell$ = {_rho:.2f}' + f'\u00b0 \u2248 {_rho_km:.0f} km  (corr \u2248 0.14)')
    _ax.axhline(0.14, color='C0', ls=':', lw=0.8, alpha=0.6)
    _ax.set_xlabel('Distance $r$ (km)')
    _ax.set_ylabel('Correlation $C(r)$')
    _ax.set_title('Matern v=1 correlation function')
    _ax.legend(fontsize=9)
    _ax.set_ylim(-0.02, 1.05)
    _ax.set_xlim(0, None)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(fm2d_grid_shape, np):
    # Starting model: uniform 3 km/s (slowness in s/km for pyfm2d)
    _n_params = fm2d_grid_shape[0] * fm2d_grid_shape[1]
    m0_pyfm2d = np.full(_n_params, 1.0 / 3.0)  # 3 km/s -> 1/3 s/km
    return (m0_pyfm2d,)


@app.cell
def _(
    cofi,
    ell,
    fm2d_grid_shape,
    grid_resolution,
    m0_pyfm2d,
    sigma,
):
    _n_lon, _n_lat = fm2d_grid_shape
    _n_params = _n_lon * _n_lat

    # SPDE Matern v=1 via cofi.utils. Uses Neumann (zero-flux) BCs, which
    # avoids spurious boundary anchoring present in plain tridiagonal truncation.
    regularization_pyfm2d = cofi.utils.SPDEMaternReg(
        model_shape=fm2d_grid_shape,
        ell=ell,
        sigma=sigma,
        grid_spacing=grid_resolution,
        reference_model=m0_pyfm2d,
    )
    _ell_km = ell * 111.0
    reg_label = (
        "SPDE Matern v=1 via cofi.utils "
        f"(ell={ell:.2f} deg ~ {_ell_km:.0f} km, rho=2ell~{2*ell:.2f} deg, sigma={sigma} s/km)"
    )

    R_pyfm2d = regularization_pyfm2d.matrix
    print(f"Prior regularization ({reg_label}):")
    print(f"  Grid: {_n_lon} lon x {_n_lat} lat = {_n_params:,} cells")
    print(f"  R shape: {R_pyfm2d.shape}")
    return (R_pyfm2d,)


@app.cell
def _(ccrs, np, plt, scm):
    # Set up map projection and boundaries for plotting
    proj = ccrs.LambertConformal(
        central_longitude=135,
        central_latitude=-27,
        cutoff=80,
        standard_parallels=(-18, -36)
    )
    transform = ccrs.PlateCarree()
    map_boundaries = [113, 153, -45, -8]

    def plot_map(phase_velocity, title=None, grid_shape=None, extent=None, vmin=None, vmax=None):
        """Plot a phase-velocity map on the pyfm2d grid."""
        fig = plt.figure(figsize=(5, 6.5))
        ax = plt.subplot(111, projection=proj)
        ax.coastlines()
        _lon = np.linspace(extent[0], extent[1], grid_shape[0])
        _lat = np.linspace(extent[2], extent[3], grid_shape[1])
        _lon_grid, _lat_grid = np.meshgrid(_lon, _lat)
        img = ax.pcolormesh(_lon_grid, _lat_grid, phase_velocity.reshape(grid_shape).T,
                            cmap=scm.roma, transform=transform, vmin=vmin, vmax=vmax)
        cb = fig.colorbar(img, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05)
        cb.set_label('Phase velocity [km/s]')
        ax.set_extent(map_boundaries, crs=transform)
        if title:
            ax.set_title(title)
        plt.tight_layout()
        return fig

    return (plot_map,)


@app.cell
def _():
    cmin = 2.1  # minimum velocity for colourbar
    cmax = 3.8  # maximum velocity for colourbar
    return cmax, cmin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gaussian Variational Inference

    The prior precision $\mathbf{Q} = \mathbf{R}^\top\mathbf{R}$ uses the SPDE Matern
    operator directly — no additional scaling factor. The physically meaningful $\sigma$
    ensures the prior is correctly calibrated.

    The data covariance $\mathbf{C}_d = \sigma_d^2 \mathbf{I}$ controls how strongly
    the data pulls the solution away from the prior. A two-pass scheme is used:

    1. **Pass 1**: Estimate $\sigma_d$ from the residual at $m_0$ (rough), run MAP-only
       to find a good solution.
    2. **Pass 2**: Re-estimate $\sigma_d$ from the residual at the MAP (accurate),
       then run the full VI with the updated $\mathbf{C}_d$.

    This ensures $\sigma_d$ reflects the true data misfit, not the gap between $m_0$
    and the data, giving a properly calibrated data-vs-prior balance.
    """)
    return


@app.cell
def _(
    Inversion,
    R_pyfm2d,
    d_obs_tt,
    forward_pyfm2d,
    inv_options,
    inv_problem,
    m0_pyfm2d,
    np,
    scipy,
):
    _n_obs = len(d_obs_tt)

    # Non-augmented forward and Jacobian for VI
    def vi_forward(m):
        return forward_pyfm2d(m, progress=False)

    def vi_jacobian(m):
        _, G = forward_pyfm2d(m, return_jacobian=True, progress=False)
        return G

    # Prior precision Q = R.T @ R (SPDE Matern, sparse)
    # No mu scaling — sigma is physically meaningful, C_d controls the balance
    Q_prior = (R_pyfm2d.T @ R_pyfm2d).tocsc()

    # --- Pass 1: rough sigma_d from residual at m0, run MAP only ---
    _d_pred_m0 = forward_pyfm2d(m0_pyfm2d, progress=False)
    sigma_d_m0 = np.std(d_obs_tt - _d_pred_m0)
    print(f"Pass 1: rough sigma_d at m0 = {sigma_d_m0:.4f} s")

    _Cd_inv_m0 = scipy.sparse.diags(
        [1.0 / sigma_d_m0**2], [0], shape=(_n_obs, _n_obs), format='csc'
    )

    inv_problem.set_data(d_obs_tt)
    inv_problem.set_initial_model(m0_pyfm2d)
    inv_problem.set_forward(vi_forward)
    inv_problem.set_jacobian(vi_jacobian)
    inv_problem.set_data_covariance_inv(_Cd_inv_m0)

    inv_options.set_solving_method("variational inference")
    inv_options.set_tool("cofi.gaussian_vi")
    inv_options.set_params(
        prior_precision=Q_prior,
        prior_mean=m0_pyfm2d,
        num_iterations=0,            # MAP only
        num_samples=1,
        map_num_iterations=10,
        learning_rate_mean=0.02,
        learning_rate_precision=0.05,
        random_seed=42,
        verbose=True,
    )

    print("Running Pass 1 (MAP only)...")
    print("-" * 60)
    _pass1 = Inversion(inv_problem, inv_options).run()
    print("-" * 60)
    m_map = _pass1.map_model

    # --- Pass 2: update sigma_d from residual at MAP, run full VI ---
    _d_pred_map = forward_pyfm2d(m_map, progress=False)
    sigma_d = np.std(d_obs_tt - _d_pred_map)
    print(f"\nPass 2: updated sigma_d at MAP = {sigma_d:.4f} s")

    Cd_inv = scipy.sparse.diags(
        [1.0 / sigma_d**2], [0], shape=(_n_obs, _n_obs), format='csc'
    )

    inv_problem.set_initial_model(m_map)
    inv_problem.set_data_covariance_inv(Cd_inv)

    inv_options.set_params(
        prior_precision=Q_prior,
        prior_mean=m0_pyfm2d,
        num_iterations=10,
        num_samples=4,
        map_num_iterations=5,        # MAP already near-converged from Pass 1
        learning_rate_mean=0.02,
        learning_rate_precision=0.05,
        random_seed=42,
        verbose=True,
    )

    print("Running Pass 2 (full VI with updated C_d)...")
    print("-" * 60)
    vi_result = Inversion(inv_problem, inv_options).run()
    print("-" * 60)

    c_vi = 1.0 / vi_result.model
    vi_elbo = vi_result.elbo_history
    vi_sampler = vi_result.sampler
    m_map_vi = vi_result.map_model

    print(f"\nGaussian VI result:")
    print(f"  Velocity range: [{c_vi.min():.2f}, {c_vi.max():.2f}] km/s")
    print(f"  MAP velocity range: [{(1.0/m_map_vi).min():.2f}, {(1.0/m_map_vi).max():.2f}] km/s")
    print(f"  Final ELBO: {vi_elbo[-1]:.2f}")
    return c_vi, vi_elbo, vi_sampler


@app.cell
def _(np, plt, vi_elbo):
    # ELBO convergence plot
    fig_elbo, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.plot(np.arange(1, len(vi_elbo) + 1), vi_elbo, 'k-', lw=1.2)
    _ax.set_xlabel('VI iteration')
    _ax.set_ylabel('ELBO')
    _ax.set_title('Gaussian VI convergence')
    fig_elbo.tight_layout()
    fig_elbo
    return


@app.cell
def _(c_vi, cmax, cmin, fm2d_extent, fm2d_grid_shape, plot_map):
    fig_vi_mean = plot_map(c_vi, title='Gaussian VI Posterior Mean',
                           grid_shape=fm2d_grid_shape, extent=fm2d_extent, vmin=cmin, vmax=cmax)
    fig_vi_mean
    return


@app.cell
def _(ccrs, fm2d_extent, fm2d_grid_shape, np, plt, scm, vi_sampler):
    # Draw posterior samples and compute uncertainty
    _samples = vi_sampler.sample(200)  # (200, n_params) in slowness
    _vel_samples = 1.0 / _samples      # convert to velocity
    _vel_std = np.std(_vel_samples, axis=0)

    # Plot uncertainty map
    _proj = ccrs.LambertConformal(
        central_longitude=135, central_latitude=-27,
        cutoff=80, standard_parallels=(-18, -36),
    )
    _transform = ccrs.PlateCarree()
    _lon = np.linspace(fm2d_extent[0], fm2d_extent[1], fm2d_grid_shape[0])
    _lat = np.linspace(fm2d_extent[2], fm2d_extent[3], fm2d_grid_shape[1])
    _lon_grid, _lat_grid = np.meshgrid(_lon, _lat)

    fig_vi_std = plt.figure(figsize=(5, 6.5))
    _ax = plt.subplot(111, projection=_proj)
    _ax.coastlines()
    _img = _ax.pcolormesh(
        _lon_grid, _lat_grid, _vel_std.reshape(fm2d_grid_shape).T,
        cmap=scm.bilbao, transform=_transform, shading='auto',
    )
    _cb = fig_vi_std.colorbar(_img, ax=_ax, orientation='horizontal', shrink=0.8, pad=0.05)
    _cb.set_label('Posterior std deviation [km/s]')
    _ax.set_extent([113, 153, -45, -8], crs=_transform)
    _ax.set_title('Gaussian VI Posterior Uncertainty')
    plt.tight_layout()
    fig_vi_std
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Results

    The Gaussian VI posterior mean provides a velocity map comparable to the deterministic
    nonlinear inversions (see `researcher.py`), but now accompanied by pixel-wise
    uncertainty estimates. The uncertainty map highlights regions where the data provides
    less constraint — typically at domain edges and where ray coverage is sparse.

    ### Limitations

    The Gaussian approximation assumes a unimodal, symmetric posterior. It cannot capture
    multi-modality, skewness, or the variable-complexity parameterisation that a
    trans-dimensional approach provides. For fully nonparametric uncertainty estimates,
    see the Bayesian sampling section in `researcher.py`.
    """)
    return


@app.cell
def _():
    # Create BaseProblem and InversionOptions (used by the VI cell above)
    from cofi import BaseProblem as _BP, InversionOptions as _IO
    inv_problem = _BP()
    inv_options = _IO()
    return inv_options, inv_problem


@app.cell(hide_code=True)
def _(scratch_dir):
    # Clean up the worker module temp directory
    import shutil

    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    return


@app.cell(hide_code=True)
def _(mo):
    import sys as _sys
    import importlib.metadata as _importlib_metadata
    from datetime import datetime as _datetime

    # Packages explicitly imported in this notebook
    _packages = [
        "h5py",
        "numpy",
        "scipy",
        "matplotlib",
        "cmcrameri",
        "cartopy",
        "cofi",
        "scikit-sparse",
        "marimo",
        "pyfm2d",
    ]

    _env_info = [{"Package": "Python", "Version": _sys.version.split()[0]}]
    for pkg in _packages:
        try:
            ver = _importlib_metadata.version(pkg)
            _env_info.append({"Package": pkg, "Version": ver})
        except _importlib_metadata.PackageNotFoundError:
            pass

    mo.vstack([
        mo.md(f"**Environment** — {_datetime.now().strftime('%Y-%m-%d %H:%M')}"),
        mo.ui.table(_env_info, selection=None, page_size=len(_env_info))
    ])
    return


if __name__ == "__main__":
    app.run()
