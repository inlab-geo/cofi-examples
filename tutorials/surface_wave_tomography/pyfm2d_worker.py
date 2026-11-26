"""Worker function for parallel pyfm2d ray tracing.

This module lives next to the notebook because ProcessPoolExecutor
requires picklable functions defined at module level.
"""


def compute_single_source(args):
    """Compute travel times for a single source using pyfm2d.

    Parameters
    ----------
    args : tuple
        (src_idx, velocity, receivers, source, extent, return_jacobian)

    Returns
    -------
    tuple
        (src_idx, ttimes, frechet_data) where frechet_data is tuple for pickling
    """
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
