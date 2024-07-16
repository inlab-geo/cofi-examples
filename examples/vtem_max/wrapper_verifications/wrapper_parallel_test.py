from wrapper_main import *


transmitters_setup = {
    "tx": numpy.array([190., 205., 220.]),                  # transmitter easting/x-position
    "ty": numpy.array([100., 100., 100.]),                  # transmitter northing/y-position
    "tz": numpy.array([50., 50., 50.]),                     # transmitter height/z-position
    "tazi": numpy.deg2rad(numpy.array([90., 90., 90.])),    # transmitter azimuth
    "tincl": numpy.deg2rad(numpy.array([6., 6., 6.])),      # transmitter inclination
    "rx": numpy.array([190., 205., 220.]),                  # receiver easting/x-position
    "ry": numpy.array([100., 100., 100.]),                  # receiver northing/y-position
    "rz": numpy.array([50., 50., 50.]),                     # receiver height/z-position
    "trdx": numpy.array([0., 0., 0.]),                      # transmitter receiver separation inline
    "trdy": numpy.array([0., 0., 0.]),                      # transmitter receiver separation crossline
    "trdz": numpy.array([0., 0., 0.]),                      # transmitter receiver separation vertical
}

transmitter_setup_1 = {k: numpy.array([v[0]]) for k, v in transmitters_setup.items()}
transmitter_setup_2 = {k: numpy.array([v[1]]) for k, v in transmitters_setup.items()}
transmitter_setup_3 = {k: numpy.array([v[2]]) for k, v in transmitters_setup.items()}


# ------ sequential computing
forward_1 = ForwardWrapper(true_model, problem_setup, system_spec, transmitter_setup_1, survey_data, ["pdip"])
forward_2 = ForwardWrapper(true_model, problem_setup, system_spec, transmitter_setup_2, survey_data, ["pdip"])
forward_3 = ForwardWrapper(true_model, problem_setup, system_spec, transmitter_setup_3, survey_data, ["pdip"])

model = numpy.array([60])

dpred_1 = forward_1(model)
dpred_2 = forward_2(model)
dpred_3 = forward_3(model)


# ------ parallel computing
forward_all = ForwardWrapper(true_model, problem_setup, system_spec, transmitters_setup, survey_data, ["pdip"])
dpred = forward_all(model)
