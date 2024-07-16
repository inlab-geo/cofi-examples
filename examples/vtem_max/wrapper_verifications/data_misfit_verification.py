from wrapper_main import *


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
true_model = numpy.array([60])
true_dpred = forward(true_model)

def data_misfit(model):
    dpred = forward(model)
    return numpy.linalg.norm(dpred - true_dpred)

def data_misfit_gradient(model):
    dpred = forward(model)
    jacobian = forward.jacobian(model)
    return 2 * jacobian.T @ (dpred - true_dpred)


all_models = [numpy.array([pdip]) for pdip in range(40, 140, 5)]
all_misfits = []
all_gradients = []
for model in all_models:
    misfit = data_misfit(model)
    gradient = data_misfit_gradient(model)
    all_misfits.append(misfit)
    all_gradients.append(gradient)
    print(f"pdip: {model}, data misfit: {misfit}, gradient: {gradient}")


fig, ax = plt.subplots()
ax.plot(all_models, all_misfits)
ax.plot(all_models, all_gradients, c="green")
ax.set_xlabel("pdip")
ax.set_ylabel("data misfit")
fig.savefig("data_misfit_verification.png")
