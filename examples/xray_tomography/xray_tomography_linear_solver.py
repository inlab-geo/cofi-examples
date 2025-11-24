"""Xray tomography problem solved with CoFI linear system solver,
with data uncertainty and regularization taken into account.
"""

import numpy as np
import matplotlib.pyplot as plt

from cofi import BaseProblem, InversionOptions, Inversion
from cofi.utils import QuadraticReg
import xrayTomography as xrt

# setup
loaded_dict = np.load('../../data/travel_time_tomography/linear_tomo_example.npz')
linear_tomo_example = dict(loaded_dict)
loaded_dict.close()
paths = linear_tomo_example["_paths"]
data = linear_tomo_example["_attns"]
data_size = len(data)
starting_model = linear_tomo_example["_start"]
model_size,model_shape = starting_model.size,starting_model.shape
good_model = linear_tomo_example["_true"]

# forward model
attns, jacobian = xrt.tracer(starting_model,paths)

# define CoFI BaseProblem
xrt_problem = BaseProblem()
xrt_problem.set_data(data)
xrt_problem.set_jacobian(jacobian)
sigma = 0.002
lamda = 50
data_cov_inv = np.identity(data_size) * (1 / sigma**2)
xrt_problem.set_data_covariance_inv(data_cov_inv)
xrt_problem.set_regularization(lamda * QuadraticReg(model_shape=(model_size,)))

# define CoFI InversionOptions
my_options = InversionOptions()
my_options.set_tool("scipy.linalg.lstsq")

# define CoFI Inversion and run it
inv = Inversion(xrt_problem, my_options)
inv_result = inv.run()
inv_result.summary()

# plot inferred model
xrt.displayModel(inv_result.model.reshape(model_shape),
                 clim=(1, 1.5),
                 cmap=plt.cm.Blues,
                 title='Recovered model',
                 filename="xray_tomography_inferred_model"); # inferred model

# plot true model
xrt.displayModel(good_model,
                 clim=(1, 1.5),
                 cmap=plt.cm.Blues,
                 title='True model',
                 filename="xray_tomography_true_model"); # true model
