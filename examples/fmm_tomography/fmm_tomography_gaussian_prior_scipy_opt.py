"""Seismic waveform inference with Fast Marching and CoFI

Based on original scripts by Andrew Valentine & Malcolm Sambridge -
Research School of Earth Sciences, The Australian National University
Last updated July 2022

"""

import numpy as np

from cofi import BaseProblem, InversionOptions, Inversion
from cofi.utils import GaussianPrior
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


# define regularization (Gaussian Prior)
corrx = 3.0
corry = 3.0
#sigma_slowness = 0.002**2
sigma_slowness = 2.5E-6
gauss_weight = 0.01
gaussian_prior = gauss_weight * GaussianPrior(
    model_covariance_inv=((corrx, corry), sigma_slowness),
    mean_model=ref_start_slowness.reshape(model_shape)
)

# define data covariance matrix
sigma = 0.000008          # data standard deviation of noise
Cd = np.zeros([data_size, data_size])
np.fill_diagonal(Cd, sigma**2)
Cdi = np.zeros([data_size, data_size])
np.fill_diagonal(Cdi, 1 / sigma**2)

# define chi square function
def chi_square(model_slowness, obstimes, Cd_inv):
    options = wt.WaveTrackerOptions(cartesian=True)
    result = wt.calc_wavefronts(1./model_slowness.reshape(model_shape),receivers,sources,extent=extent,options=options) # track wavefronts
    pred = result.ttimes   
    residual = obstimes - pred
    return residual.T @ Cd_inv @ residual + gaussian_prior(model_slowness)


def gradient(model_slowness, obstimes, Cd_inv):
    options = wt.WaveTrackerOptions(
        paths=True,
        frechet=True,
        cartesian=True,
        )
    result = wt.calc_wavefronts(1./model_slowness.reshape(model_shape),receivers,sources,extent=extent,options=options) # track wavefronts
    pred = result.ttimes
    jac = result.frechet.toarray()
        
    residual = obstimes - pred
    return -jac.T @ Cd_inv @ residual + gaussian_prior.gradient(model_slowness)


def hessian(model_slowness, Cd_inv):
    options = wt.WaveTrackerOptions(
        paths=True,
        frechet=True,
        cartesian=True,
        )
    result = wt.calc_wavefronts(1./model_slowness.reshape(model_shape),receivers,sources,extent=extent,options=options)
    A = result.frechet.toarray()
    return A.T @ Cd_inv @ A + gaussian_prior.hessian(model_slowness)


# define CoFI BaseProblem
fmm_problem = BaseProblem()
fmm_problem.set_initial_model(ref_start_slowness.flatten())
fmm_problem.set_objective(chi_square, args=[obstimes, Cdi])
fmm_problem.set_gradient(gradient, args=[obstimes, Cdi])
fmm_problem.set_hessian(hessian, args=[Cdi])

# define CoFI InversionOptions
inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.minimize")
method = "Newton-CG"
inv_options.set_params(method=method, verbose=True, options={"xtol": 1e-12})

# define CoFI Inversion and run
inv = Inversion(fmm_problem, inv_options)
inv_result = inv.run()
vmodel_inverted = 1./inv_result.model.reshape(model_shape)
wt.display_model(vmodel_inverted,extent=extent,title='Recovered model',filename=f"figs/fmm_gaussian_prior_scipy_{method}") # inverted model

# Plot the true model
wt.display_model(good_model,extent=extent,title='True model',filename="figs/fmm_true_model") # true model
