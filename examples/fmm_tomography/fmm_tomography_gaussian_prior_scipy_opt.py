"""Seismic waveform inference with Fast Marching and CoFI

Based on original scripts by Andrew Valentine & Malcolm Sambridge -
Research School of Earth Sciences, The Australian National University
Last updated July 2022

"""

import numpy as np

from cofi import BaseProblem, InversionOptions, Inversion
from cofi.utils import GaussianPrior
from espresso import FmmTomography
import pyfm2d as wt # import fmm package 

usepyfm2d = True # switch to use either fmm from pyfm2d (True) package or geo-espresso (False)

# get espresso problem FmmTomography information
fmm = FmmTomography()

# temporarily overwrite espresso data
read = True
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
    
model_size = fmm.model_size
model_shape = fmm.model_shape
data_size = fmm.data_size
ref_start_slowness = fmm.starting_model

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
def chi_square(model_slowness, obstimes, esp_fmm, Cd_inv):
    if(usepyfm2d):
        options = wt.WaveTrackerOptions(
                  cartesian=True,
                  )
        result = wt.calc_wavefronts(1./model_slowness.reshape(fmm.model_shape),receivers,sources,extent=fmm.extent,options=options) # track wavefronts
        pred = result.ttimes   
    else:
        pred = esp_fmm.forward(model_slowness)
    residual = obstimes - pred
    model_diff = model_slowness - ref_start_slowness
    return residual.T @ Cd_inv @ residual + gaussian_prior(model_slowness)


def gradient(model_slowness, obstimes, esp_fmm, Cd_inv):
    if(usepyfm2d):
        options = wt.WaveTrackerOptions(
                    paths=True,
                    frechet=True,
                    cartesian=True,
                    )
        result = wt.calc_wavefronts(1./model_slowness.reshape(fmm.model_shape),receivers,sources,extent=fmm.extent,options=options) # track wavefronts
        pred = result.ttimes
        jac = result.frechet.toarray()
    else:
        pred, jac = esp_fmm.forward(model_slowness, return_jacobian=True)
        
    residual = obstimes - pred
    model_diff = model_slowness - ref_start_slowness
    return -jac.T @ Cd_inv @ residual + gaussian_prior.gradient(model_slowness)


def hessian(model_slowness, esp_fmm, Cd_inv):
    if(usepyfm2d):
        options = wt.WaveTrackerOptions(
                    paths=True,
                    frechet=True,
                    cartesian=True,
                    )
        result = wt.calc_wavefronts(1./model_slowness.reshape(fmm.model_shape),receivers,sources,extent=fmm.extent,options=options)
        A = result.frechet.toarray()
    else:
        A = esp_fmm.jacobian(model_slowness)     
    return A.T @ Cd_inv @ A + gaussian_prior.hessian(model_slowness)


# define CoFI BaseProblem
fmm_problem = BaseProblem()
fmm_problem.set_initial_model(ref_start_slowness)
fmm_problem.set_objective(chi_square, args=[obstimes,fmm, Cdi])
fmm_problem.set_gradient(gradient, args=[obstimes,fmm, Cdi])
fmm_problem.set_hessian(hessian, args=[fmm, Cdi])

# define CoFI InversionOptions
inv_options = InversionOptions()
inv_options.set_tool("scipy.optimize.minimize")
method = "Newton-CG"
inv_options.set_params(method=method, options={"xtol": 1e-12})

# define CoFI Inversion and run
inv = Inversion(fmm_problem, inv_options)
inv_result = inv.run()
ax1 = fmm.plot_model(inv_result.model)
ax1.get_figure().savefig(f"figs/fmm_gaussian_prior_scipy_{method}")

# Plot the true model
ax2 = fmm.plot_model(fmm.good_model)
ax2.get_figure().savefig("figs/fmm_true_model")
