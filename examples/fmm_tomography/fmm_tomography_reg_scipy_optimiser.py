"""Seismic waveform inference with Fast Marching and CoFI

Based on original scripts by Andrew Valentine & Malcolm Sambridge -
Research School of Earth Sciences, The Australian National University
Last updated July 2022

"""
import numpy as np

from cofi import BaseProblem, InversionOptions, Inversion
from cofi.utils import QuadraticReg
from espresso import FmmTomography
import pyfm2d as wt # import fmm package 

usepyfm2d = True # switch to use either fmm from pyfm2d (True) package or geo-espresso (False)

# get espresso problem FmmTomography information
fmm = FmmTomography()

# temporarily overwrite espresso data
read = False
if(read):
    data_base_path = "../../data/fmm_tomography"
    ttdat = np.loadtxt(f"{data_base_path}/ttimes_crossb_nwt_s10_r10.dat")
    sources = np.loadtxt(f"{data_base_path}/sources_crossb_nwt_s10.dat")[:,1:]
    receivers = np.loadtxt(f"{data_base_path}/receivers_crossb_nwt_r10.dat")[:,1:]
    obstimes = ttdat[:,2]
else:
    obstimes = fmm.data
    sources = fmm.sources
    receivers = fmm.receivers

model_size = fmm.model_size  # number of model parameters
model_shape = fmm.model_shape  # 2D spatial grids
data_size = fmm.data_size  # number of data points
ref_start_slowness = fmm.starting_model

# define CoFI BaseProblem
fmm_problem = BaseProblem()
fmm_problem.set_initial_model(ref_start_slowness)

# add regularization: damping + flattening + smoothing
damping_factor = 0
flattening_factor = 1e8
smoothing_factor = 1e7
reg_damping = damping_factor * QuadraticReg(
    model_shape=model_shape, 
    weighting_matrix="damping", 
    reference_model=ref_start_slowness
)
reg_flattening = flattening_factor * QuadraticReg(
    model_shape=model_shape,
    weighting_matrix="flattening"
)
reg_smoothing = smoothing_factor * QuadraticReg(
    model_shape=model_shape,
    weighting_matrix="smoothing"
)

reg = reg_damping + reg_flattening + reg_smoothing
fmm_problem.set_regularization(reg)

sigma = 0.000008          # data standard deviation of noise


def objective_func(slowness):
    if(usepyfm2d):
        options = wt.WaveTrackerOptions(
                  cartesian=True,
                  )
        result = wt.calc_wavefronts(1./slowness.reshape(fmm.model_shape),receivers,sources,extent=fmm.extent,options=options) # track wavefronts
        ttimes = result.ttimes
    else:
        ttimes = fmm.forward(slowness)
    residual = obstimes - ttimes
    data_misfit = residual.T @ residual / sigma**2
    model_reg = reg(slowness)
    return data_misfit + model_reg


def gradient(slowness):
    if(usepyfm2d):
        options = wt.WaveTrackerOptions(
                    paths=True,
                    frechet=True,
                    cartesian=True,
                    )
        result = wt.calc_wavefronts(1./slowness.reshape(fmm.model_shape),receivers,sources,extent=fmm.extent,options=options) # track wavefronts
        ttimes = result.ttimes
        A = result.frechet.toarray()
    else:
        ttimes, A = fmm.forward(slowness, with_jacobian=True)
    data_misfit_grad = -2 * A.T @ (obstimes - ttimes) / sigma**2
    model_reg_grad = reg.gradient(slowness)
    return data_misfit_grad + model_reg_grad


def hessian(slowness):
    if(usepyfm2d):
        options = wt.WaveTrackerOptions(
                    paths=True,
                    frechet=True,
                    cartesian=True,
                    )
        result = wt.calc_wavefronts(1./slowness.reshape(fmm.model_shape),receivers,sources,extent=fmm.extent,options=options)
        A = result.frechet.toarray()
    else:
        A = fmm.jacobian(slowness)
    data_misfit_hess = 2 * A.T @ A / sigma**2
    model_reg_hess = reg.hessian(slowness)
    return data_misfit_hess + model_reg_hess


fmm_problem.set_objective(objective_func)
fmm_problem.set_gradient(gradient)
fmm_problem.set_hessian(hessian)

# Define CoFI InversionOptions
inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.minimize")
inv_options.set_params(method="Newton-CG", options={"xtol": 1e-12})

# Define CoFI Inversion and run
inv_newton = Inversion(fmm_problem, inv_options)
inv_result_newton = inv_newton.run()
ax = fmm.plot_model(inv_result_newton.model)
ax.get_figure().savefig(f"figs/fmm_{int(damping_factor)}_{int(flattening_factor)}_{int(smoothing_factor)}_scipy_optimiser")

# Plot the true model
ax2 = fmm.plot_model(fmm.good_model)
ax2.get_figure().savefig("figs/fmm_true_model")
