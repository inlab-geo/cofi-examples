import numpy as np
import matplotlib.pyplot as plt
import bayesbay as bb
import cofi


# DIMENSIONS AND TRUE COEFFICIENTS
N_DIMS = 4
M0, M1, M2, M3 = 20, -10, -3, 1

# DATA AND NOISE
N_DATA = 15
DATA_X = np.linspace(-5, 10, N_DATA)
DATA_NOISE_STD = 20

# INFERENCE SETTINGS
N_CHAINS = 4

# generate synthetic data
fwd_operator = np.vander(DATA_X, N_DIMS, True)
y = fwd_operator @ [M0, M1, M2, M3]
y_noisy = y + np.random.normal(0, DATA_NOISE_STD, y.shape)

# define parameters
m0 = bb.prior.UniformPrior("m0", -100, 100, 5)
m1 = bb.prior.UniformPrior("m1", -50, 50, 5)
m2 = bb.prior.UniformPrior("m2", -20, 20, 3)
m3 = bb.prior.UniformPrior("m3", -10, 10, 2)

# define parameterization
param_space = bb.parameterization.ParameterSpace(
    name="my_param_space",
    n_dimensions=1,
    parameters=[m0, m1, m2, m3],
)
parameterization = bb.parameterization.Parameterization(param_space)

# define forward function
def my_fwd(state: bb.State) -> np.ndarray:
    m = [state["my_param_space"][f"m{i}"] for i in range(N_DIMS)]
    return np.squeeze(fwd_operator @ np.array(m))


fwd_functions = [my_fwd]

# define data target
targets = [bb.likelihood.Target("my_data", y_noisy, 1 / DATA_NOISE_STD**2)]

# Define log-likelihood
log_likelihood = bb.likelihood.LogLikelihood(
    targets=targets, fwd_functions=fwd_functions
)
# initialize walkers
walkers_start = []
for i in range(N_CHAINS):
    walkers_start.append(parameterization.initialize())

# run the sampling
inv_problem = cofi.BaseProblem()
inv_options = cofi.InversionOptions()
inv_options.set_tool("bayesbay")
inv_options.set_params(
    log_like_ratio_func=log_likelihood,
    perturbation_funcs=parameterization.perturbation_funcs,
    walkers_starting_states=walkers_start,
    n_chains=N_CHAINS,
    n_iterations=100_000,
    burnin_iterations=10_000,
    save_every=500,
    verbose=False,
)
inversion = cofi.Inversion(inv_problem, inv_options)
inv_result = inversion.run()

# get results and plot
results = inv_result.models
coefficients_samples = np.squeeze(
    np.array([results[f"my_param_space.m{i}"] for i in range(N_DIMS)])
)
fig, ax = plt.subplots()
all_y_pred = np.zeros((coefficients_samples.shape[1], len(y)))
for i, coefficients in enumerate(coefficients_samples.T):
    y_pred = fwd_operator @ coefficients
    all_y_pred[i, :] = y_pred
    if i == 0:
        ax.plot(DATA_X, y_pred, c="gray", lw=0.05, label="Predicted data from samples")
    else:
        ax.plot(DATA_X, y_pred, c="gray", lw=0.05)
ax.plot(DATA_X, y, c="orange", label="Noise-free data from true model")
ax.plot(
    DATA_X, np.median(all_y_pred, axis=0), c="blue", label="Median predicted sample"
)
ax.scatter(DATA_X, y_noisy, c="purple", label="Noisy data used for inference", zorder=3)
ax.legend()
fig.savefig("linear_regression_bayesbay")
