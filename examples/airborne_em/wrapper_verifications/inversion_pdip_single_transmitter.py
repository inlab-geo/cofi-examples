import cofi
from wrapper_main import *


# ------- set up what to invert
transmitters_setup = {
    "tx": numpy.array([205.]),                  # transmitter easting/x-position
    "ty": numpy.array([100.]),                  # transmitter northing/y-position
    "tz": numpy.array([50.]),                   # transmitter height/z-position
    "tazi": numpy.deg2rad(numpy.array([90.])),  # transmitter azimuth
    "tincl": numpy.deg2rad(numpy.array([6.])),  # transmitter inclination
    "rx": numpy.array([205.]),                  # receiver easting/x-position
    "ry": numpy.array([100.]),                  # receiver northing/y-position
    "rz": numpy.array([50.]),                   # receiver height/z-position
    "trdx": numpy.array([0.]),                  # transmitter receiver separation inline
    "trdy": numpy.array([0.]),                  # transmitter receiver separation crossline
    "trdz": numpy.array([0.]),                  # transmitter receiver separation vertical
}

forward = ForwardWrapper(true_model, problem_setup, system_spec, transmitters_setup, survey_data, ["pdip"])
true_param_value = numpy.array([60])


# ------- generate synthetic data
data_noise = 0.01
data_pred_true = forward(true_param_value)
data_obs = data_pred_true # + numpy.random.normal(0, data_noise, data_pred_true.shape)


# ------- initialise a model for inversion
init_param_value = numpy.array([45])


# ------- define helper functions
def my_objective(model):
    dpred = forward(model)
    residual = dpred - data_obs
    return residual.T @ residual

def my_gradient(model):
    dpred = forward(model)
    jacobian = forward.jacobian(model)
    residual = dpred - data_obs
    return 2 * jacobian.T @ residual

def my_hessian(model):
    jacobian = forward.jacobian(model)
    return 2 * jacobian.T @ jacobian


# ------- run inversion
my_problem = cofi.BaseProblem()
my_problem.set_objective(my_objective)
my_problem.set_gradient(my_gradient)
my_problem.set_hessian(my_hessian)
my_problem.set_initial_model(init_param_value)
my_options = cofi.InversionOptions()
my_options.set_tool("scipy.optimize.minimize")
my_options.set_params(method="Newton-CG")
my_inversion = cofi.Inversion(my_problem, my_options)
my_result = my_inversion.run()
print(my_result.model)


# -------- plot data and inferred model
figure, (ax1, ax2) = plt.subplots(1, 2)
plot_data(true_param_value, forward, "data from true model", ax1, ax2, color="purple")
plot_data(my_result.model, forward, "data from inverted model", ax1, ax2, color="red", linestyle="-.")
plot_data(init_param_value, forward, "data from init model", ax1, ax2, color="green", linestyle=":")
ax1.legend(loc="upper center")
ax2.legend(loc="upper center")
ax1.set_title("vertical")
ax2.set_title("inline")
figure.savefig("inversion_pdip_single_transmitter_data.png")

fig, axes = plt.subplots(2, 2, sharex="col", sharey="row")
axes[1,1].axis("off")
plot_plate_faces(
    "plate_true", forward, problem_setup, true_param_value, 
    axes[0,0], axes[0,1], axes[1,0], color="purple", label="true model"
)
plot_plate_faces(
    "plate_inverted", forward, problem_setup, my_result.model, 
    axes[0,0], axes[0,1], axes[1,0], color="red", label="inverted model", linestyle="dotted"
)
plot_plate_faces(
    "plate_init", forward, problem_setup, init_param_value, 
    axes[0,0], axes[0,1], axes[1,0], color="green", label="init model"
)
axes[0,0].legend()
axes[0,1].legend()
axes[1,0].legend()
fig.savefig("inversion_pdip_single_transmitter_plate_faces.png")
