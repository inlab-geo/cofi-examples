"""Seismic waveform inference with Fast Marching and CoFI

Based on original scripts by Andrew Valentine & Malcolm Sambridge -
Research School of Earth Sciences, The Australian National University
Last updated July 2022

"""
import numpy as np

from cofi import BaseProblem, InversionOptions, Inversion
from cofi.utils import QuadraticReg
import pyfm2d as wt # import fmm package 


# read in problem data
loaded_dict = np.load('../../data/travel_time_tomography/nonlinear_tomo_example.npz')
nonlinear_tomo_example = dict(loaded_dict)
loaded_dict.close()
    
# set up problem
good_model = nonlinear_tomo_example["_mtrue"]
extent = nonlinear_tomo_example["extent"]
sources = nonlinear_tomo_example["sources"]
receivers = nonlinear_tomo_example["receivers"]
obstimes = nonlinear_tomo_example["_data"]
ref_start_slowness = nonlinear_tomo_example["_sstart"] # use the starting guess supplied by the nonlinear example
model_size = good_model.size                           # number of model parameters
model_shape = good_model.shape                         # 2D spatial grid shape
data_size = data_size = len(obstimes)                  # number of data
print(' New data set have:\n',len(receivers),' receivers\n',len(sources),' sources\n',len(obstimes),' travel times\n',
'Range of travel times: ',np.min(obstimes),'to',np.max(obstimes),'\n Mean travel time:',np.mean(obstimes))


# define CoFI BaseProblem
fmm_problem = BaseProblem()
fmm_problem.set_initial_model(ref_start_slowness.flatten())

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
    options = wt.WaveTrackerOptions(cartesian=True)
    result = wt.calc_wavefronts(1./slowness.reshape(model_shape),receivers,sources,extent=extent,options=options) # track wavefronts
    ttimes = result.ttimes
    residual = obstimes - ttimes
    data_misfit = residual.T @ residual / sigma**2
    model_reg = reg(slowness)
    return data_misfit + model_reg


def gradient(slowness):
    options = wt.WaveTrackerOptions(
        paths=True,
        frechet=True,
        cartesian=True)
    result = wt.calc_wavefronts(1./slowness.reshape(model_shape),receivers,sources,extent=extent,options=options) # track wavefronts
    ttimes = result.ttimes
    A = result.frechet.toarray()
    data_misfit_grad = -2 * A.T @ (obstimes - ttimes) / sigma**2
    model_reg_grad = reg.gradient(slowness)
    return data_misfit_grad + model_reg_grad


def hessian(slowness):
    options = wt.WaveTrackerOptions(
        paths=True,
        frechet=True,
        cartesian=True,
        )
    result = wt.calc_wavefronts(1./slowness.reshape(model_shape),receivers,sources,extent=extent,options=options)
    A = result.frechet.toarray()
    data_misfit_hess = 2 * A.T @ A / sigma**2
    model_reg_hess = reg.hessian(slowness)
    return data_misfit_hess + model_reg_hess


fmm_problem.set_objective(objective_func)
fmm_problem.set_gradient(gradient)
fmm_problem.set_hessian(hessian)

# Define CoFI InversionOptions
inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.minimize")
inv_options.set_params(method="Newton-CG", verbose=True, options={"xtol": 1e-12})

# Define CoFI Inversion and run
inv_newton = Inversion(fmm_problem, inv_options)
inv_result_newton = inv_newton.run()
vmodel_inverted = 1./inv_result_newton.model.reshape(model_shape)
wt.display_model(vmodel_inverted,extent=extent,title='Recovered model',filename=f"figs/fmm_{int(damping_factor)}_{int(flattening_factor)}_{int(smoothing_factor)}_scipy_optimiser") # inverted model


# Plot the true model
wt.display_model(good_model,extent=extent,title='True model',filename="figs/fmm_true_model") # true model
