from typing import List, Dict
import warnings
import joblib
import os
import numpy
import operator
import matplotlib.pyplot as plt
import pyp223
import geom_lib


# ------- problem setup
problem_setup = {
    "nlyr": 2,  # number of layers (icl. halfspace)
    "nstat": 1,  # number of fiducials/stations
    "nplt": 1,  # number of thin plates
    "cellw": 25,  # cell width
    "pthk": numpy.array([1]),  # plates thickness
    "plng": numpy.deg2rad(numpy.array([0])),  # plates plunge (orientation)
}

# ------- system specification with waveform and gates read from file
system_spec = {
    "ncmp": 2,  # system spec: number of components
    "cmp": 2,  # system spec: active components
    "ntrn": 3,  # system spec: number of transmitter turns
    "txarea": 531,  # system spec: transmitter area
    "ampt": 0,  # system spec: amplitude type AMPS 0
}

# ------- example survey settings
tx_min = 115
tx_max = 281
tx_interval = 15
n_fiducials = (tx_max - tx_min - 1) // tx_interval + 1
tx = numpy.arange(tx_min, tx_max, tx_interval)
survey_setup = {
    "tx": tx,  # transmitter easting/x-position
    "ty": numpy.array([100] * n_fiducials),  # transmitter northing/y-position
    "tz": numpy.array([50] * n_fiducials),  # transmitter height/z-position
    "tazi": numpy.deg2rad(numpy.array([90] * n_fiducials)),  # transmitter azimuth
    "tincl": numpy.deg2rad(
        numpy.array([6] * n_fiducials)
    ),  # transmitter inclination
    "rx": tx,  # receiver easting/x-position
    "ry": numpy.array([100] * n_fiducials),  # receiver northing/y-position
    "rz": numpy.array([50] * n_fiducials),  # receiver height/z-position
    "trdx": numpy.array([0] * n_fiducials),  # transmitter receiver separation inline
    "trdy": numpy.array(
        [0] * n_fiducials
    ),  # transmitter receiver separation crossline
    "trdz": numpy.array(
        [0] * n_fiducials
    ),  # transmitter receiver separation vertical
}


# ------- read survey data
def read_gates_and_waveform(file_name="LeroiAir.cfl") -> dict:
    with open(file_name) as f:
        lines = f.readlines()
    nsx = int(lines[2].split()[1])
    nchnl = int(lines[2].split()[4])
    swx = []
    waveform = []
    for i in range(nsx):
        fields = lines[3 + i].split()
        swx.append(float(fields[0]) / 1000.0)
        waveform.append(fields[1])
    topn = []
    tcls = []
    for i in range(nchnl):
        fields = lines[3 + nsx + i].split()
        topn.append(float(fields[0]) / 1000.0)
        tcls.append(float(fields[1]) / 1000.0)
    swx = numpy.array(swx)
    waveform = numpy.array(waveform)
    topn = numpy.array(topn)
    tcls = numpy.array(tcls)
    return {"nsx": nsx,"swx": swx,"waveform": waveform},{"nchnl": nchnl,"topn": topn,"tcls": tcls}


# (topn + tcls) / 2
[gates,waveform] = read_gates_and_waveform()
system_spec.update(gates)
system_spec.update(waveform)

# ------- define true model
true_model = {
    "res": numpy.array([300, 1000]),
    "thk": numpy.array([25]),
    "peast": numpy.array([175]),
    "pnorth": numpy.array([100]),
    "ptop": numpy.array([30]),
    "pres": numpy.array([0.1]),
    "plngth1": numpy.array([100]),
    "plngth2": numpy.array([100]),
    "pwdth1": numpy.array([0.1]),
    "pwdth2": numpy.array([90]),
    "pdzm": numpy.array([90]),
    "pdip": numpy.array([60]),
}


# ------- wrap forward operator
class ForwardWrapper:

    def __init__(
        self,
        sample_model_params: Dict[str, numpy.ndarray],
        problem_setup: Dict[str, numpy.ndarray],
        system_spec: Dict[str, numpy.ndarray],
        survey_setup: Dict[str, numpy.ndarray],
        params_to_invert: List[str] = None,
        data_returned: List[str] = ["vertical", "inline"],
    ):
        self.sample_model_params = sample_model_params
        self.problem_setup = problem_setup
        self.system_spec = system_spec

        print(params_to_invert)

        self.n_fiducials = survey_setup["tx"].size
        if self.n_fiducials == 1:
            self.survey_setup = survey_setup
        else:
            self.survey_setup = []
            for i in range(self.n_fiducials):
                self.survey_setup.append(
                    {k: numpy.array([v[i]]) for k, v in survey_setup.items()}
                )

        if params_to_invert is None:
            params_to_invert = sorted(self.sample_model_params.keys())
        else:
            for p in params_to_invert:
                if p not in self.sample_model_params:
                    raise ValueError(f"Invalid parameter name: {p}")
        self.params_to_invert = sorted(params_to_invert)

        self.data_returned = []
        for d in data_returned:
            if d not in ["vertical", "inline"]:
                raise ValueError(
                    f"Invalid data return type: {d}. Must be either 'inline' or 'vertical'"
                )
            self.data_returned.append(d)
        if len(self.data_returned) == 2:
            self.data_returned = [
                "vertical",
                "inline",
            ]  # make sure the order is correct

        self._init_param_length()
        self.leroiair = pyp223.LeroiAir()

    def __call__(
        self, model: numpy.ndarray, return_lengths: bool = False
    ) -> numpy.ndarray:
        
        model_dict = self.model_dict(model)
        #model_dict=model
        model_dict = {
            k: numpy.deg2rad(v) if k in ["pdzm", "pdip"] else v
            for k, v in model_dict.items()
        }
        pbres = model_dict["res"][-1]
        leroiair_failure_count = 0
        xmodl = numpy.zeros([self.system_spec["nchnl"] * self.system_spec["ncmp"]])
        if self.n_fiducials == 1:
            dpred = self._call_forward(
                pbres,
                leroiair_failure_count,
                xmodl,
                model_dict,
                self.survey_setup,
            )
        else:
            # https://joblib.readthedocs.io/en/stable/parallel.html#thread-based-parallelism-vs-process-based-parallelism
            # calling compiled extension so using `prefer="threads"`
            dpred_all = joblib.Parallel(n_jobs=self.n_fiducials, prefer="threads")(
                joblib.delayed(self._call_forward)(
                    pbres, leroiair_failure_count, xmodl, model_dict, survey_setup
                )
                for survey_setup in self.survey_setup
            )
            dpred = numpy.concatenate(dpred_all)
            if return_lengths:
                return dpred, [len(d) for d in dpred_all]
        return dpred

    def _call_forward(self, pbres, failure_count, xmodl, model_dict, survey_setup):
        dpred = self.leroiair.formod_vtem_max_data(
            pbres=pbres,
            leroiair_failure_count=failure_count,
            xmodl=xmodl,
            **model_dict,
            **{k: v for k, v in survey_setup.items() if "id" not in k},
            **self.problem_setup,
            **self.system_spec,
        )
        if len(self.data_returned) == 2:
            dpred = dpred.reshape(-1)
        elif "vertical" in self.data_returned:
            dpred = dpred[:, 0]
        else:
            dpred = dpred[:, 1]
        dpred[dpred < 0.0001] = 0.0001
        return numpy.log(dpred)

    def jacobian(self, model: numpy.ndarray, relative_step=0.1) -> numpy.ndarray:
        original_dpred = self.__call__(model)
        jac = numpy.zeros((len(original_dpred), len(model)))
        for i in range(len(model)):
            perturbed_model = model.copy()
            step = relative_step * model[i]
            perturbed_model[i] += step
            perturbed_dpred = self.__call__(perturbed_model)
            derivative = (perturbed_dpred - original_dpred) / step
            jac[:, i] = derivative
        return jac

    def _init_param_length(self):
        nlyr = self.problem_setup["nlyr"]
        nstat = self.problem_setup["nstat"]
        nplt = self.problem_setup["nplt"]
        self.param_length = {
            "res": nlyr,
            "thk": (nlyr - 1) * nstat,
            "peast": nplt,
            "pnorth": nplt,
            "ptop": nplt,
            "pres": nplt,
            "plngth1": nplt,
            "plngth2": nplt,
            "pwdth1": nplt,
            "pwdth2": nplt,
            "pdzm": nplt,
            "pdip": nplt,
        }

    def model_dict(self, model: numpy.ndarray) -> Dict[str, numpy.ndarray]:
        model_dict = dict(self.sample_model_params)

        i = 0
        for p in self.params_to_invert:
            try:
                model_dict[p] = model[i : i + self.param_length[p]]
            except IndexError:
                raise ValueError(
                    f"Invalid model length. Expected {sum(self.param_length)} in "
                    f"total for parameters: {self.params_to_invert}"
                )
            i += self.param_length[p]
        return model_dict

    def model_vector(self, model: Dict[str, numpy.ndarray]) -> numpy.ndarray:
        return numpy.concatenate([model[p] for p in self.params_to_invert])


# ------- wrap plotting functions
def plot_predicted_data(model, forward, label, ax1=None, ax2=None, **kwargs):
    vertical_returned = "vertical" in forward.data_returned
    inline_returned = "inline" in forward.data_returned
    if ax1 is None:
        if vertical_returned and inline_returned:
            _, (ax1, ax2) = plt.subplots(1, 2)
        else:
            _, ax1 = plt.subplots(1, 1)
    x = (forward.system_spec["topn"] + forward.system_spec["tcls"]) / 2
    if forward.n_fiducials == 1:
        data = forward(model)
        _plot_data(
            x, data, vertical_returned, inline_returned, ax1, ax2, label=label, **kwargs
        )
    else:
        data, data_lengths = forward(model, return_lengths=True)
        i = 0
        for length in data_lengths:
            current_data = data[i : i + length]
            if i == 0:
                _plot_data(
                    x,
                    current_data,
                    vertical_returned,
                    inline_returned,
                    ax1,
                    ax2,
                    label=label,
                    **kwargs,
                )
            else:
                _plot_data(
                    x,
                    current_data,
                    vertical_returned,
                    inline_returned,
                    ax1,
                    ax2,
                    **kwargs,
                )
            i += length


def plot_field_data(data_x, data_obs, label, ax, **kwargs):
    if len(data_obs) != len(data_x):  # there are more than one transmitters
        n_fiducials = len(data_obs) // len(data_x)
        data_length = len(data_x)
        for i in range(n_fiducials):
            data_x_transmitter = data_x
            data_obs_transmitter = data_obs[i * data_length : (i + 1) * data_length]
            if i == 0:
                _plot_data(
                    data_x_transmitter,
                    data_obs_transmitter,
                    True,
                    False,
                    ax,
                    None,
                    label=label,
                    **kwargs,
                )
            else:
                _plot_data(
                    data_x_transmitter,
                    data_obs_transmitter,
                    True,
                    False,
                    ax,
                    None,
                    **kwargs,
                )
    else:
        _plot_data(data_x, data_obs, True, False, ax, None, label=label, **kwargs)


def _plot_data(
    x,
    data,
    vertical_returned,
    inline_returned,
    ax1,
    ax2,
    xlabel="mid time of gates",
    ylabel="(pT/s)",
    **kwargs,
):
    if vertical_returned and inline_returned:
        data = data.reshape(-1, 2)
        data = numpy.exp(data)
        vertical = data[:, 0]
        inline = data[:, 1]
        ax1.loglog(x, vertical, **kwargs)
        ax2.loglog(x, inline, **kwargs)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
    else:
        ax1.loglog(x, data, **kwargs)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)


def plot_synth_vertical_vs_tx(
    model,
    forward,
    label,
    gate_idx: list = None,
    line_id: list = None,
    ax=None,
    **kwargs,
):
    if forward.n_fiducials == 1:
        raise ValueError("This function is only for multiple transmitters")
    if ax is None:
        _, ax = plt.subplots(1, 1)
    x = numpy.array(
        [forward.survey_setup[i]["tx"] for i in range(forward.n_fiducials)]
    )
    old_data_returned = forward.data_returned
    forward.data_returned = ["vertical"]
    data, data_lengths = forward(model, return_lengths=True)
    idx_to_draw = range(data_lengths[0]) if gate_idx is None else gate_idx
    if line_id is not None:
        idx_to_draw_line_id = [
            i
            for i in range(forward.n_fiducials)
            if forward.survey_setup[i]["line_id"]
            in line_id
        ]
        x = x[idx_to_draw_line_id]
    labeled = False
    for i in idx_to_draw:
        y = numpy.array([data[j] for j in range(i, len(data), data_lengths[0])])
        if line_id is not None:
            y = y[idx_to_draw_line_id]
        if not labeled:
            _plot_data(
                x,
                y,
                True,
                False,
                ax,
                None,
                "horizontal distance (m)",
                "vertical component (fT)",
                label=label,
                **kwargs,
            )
            labeled = True
        else:
            _plot_data(
                x,
                y,
                True,
                False,
                ax,
                None,
                "horizontal distance (m)",
                "vertical component (fT)",
                **kwargs,
            )
    forward.data_returned = old_data_returned


def plot_field_vertical_vs_tx(
    survey_setup,
    data_obs,
    label,
    gate_idx: list = None,
    line_id: list = None,
    annotate_fiducial_id: bool = False,
    ax=None,
    **kwargs,
):
    if survey_setup["tx"].size == 1:
        raise ValueError("This function is only for multiple transmitters")
    if ax is None:
        _, ax = plt.subplots(1, 1)
    survey_setup, _, data_obs = get_subset_data_from_gateidx_n_lineid(
        survey_setup, None, data_obs, gate_idx, line_id
    )
    x = survey_setup["tx"]
    n_gates = data_obs.size // x.size
    labled = False
    for i in range(n_gates):
        y = data_obs[i::n_gates]
        if not labled:
            _plot_data(
                x,
                y,
                True,
                False,
                ax,
                None,
                "horizontal distance (m)",
                "vertical component (fT)",
                label=label,
                **kwargs,
            )
            labled = True
            if annotate_fiducial_id:
                for j in range(0, x.size, 200):
                    ax.annotate(str(survey_setup["fiducial_id"][j]), (x[j], y[j]), fontsize=8)
        else:
            _plot_data(
                x,
                y,
                True,
                False,
                ax,
                None,
                "horizontal distance (m)",
                "vertical component (fT)",
                **kwargs,
            )

def get_subset_data_from_gateidx_n_lineid(
    survey_setup,
    survey_data,
    data_obs,
    gate_idx=None,
    line_id=None,
    transmitter_fiducial_id=None,
):
    x = survey_setup["tx"]
    n_gates_total = data_obs.size // x.size
    data_obs = data_obs.reshape((x.size, n_gates_total))
    if gate_idx is None:
        gate_idx = range(n_gates_total)
    if transmitter_fiducial_id is not None:
        if line_id is not None:
            warnings.warn(
                "Both line_id and transmitter_fiducial_id are provided. "
                "Using transmitter_fiducial_id."
            )
        transmitter_idx = [
            i
            for i in range(x.size)
            if survey_setup["fiducial_id"][i] in transmitter_fiducial_id
        ]
    else:
        if line_id is None:
            line_id = list(set(survey_setup["line_id"]))
        transmitter_idx = [
            i
            for i in range(x.size)
            if survey_setup["line_id"][i] in line_id
        ]
    new_survey_setup = {
        k: v[transmitter_idx] for k, v in survey_setup.items()
    }
    if survey_data is None:
        new_survey_data = None
    else:
        new_survey_data = dict(survey_data)
        new_survey_data["nchnl"] = len(gate_idx)
        new_survey_data["topn"] = survey_data["topn"][gate_idx]
        new_survey_data["tcls"] = survey_data["tcls"][gate_idx]
    new_data_obs = numpy.zeros(
        (
            len(transmitter_idx),
            len(gate_idx),
        )
    )
    for i, idx in enumerate(gate_idx):
        new_data_obs[:, i] = data_obs[transmitter_idx, idx]
    new_data_obs = new_data_obs.flatten()
    return new_survey_setup, new_survey_data, new_data_obs


def gmt_plate_faces(fpt, forward, problem_setup, model, surface_elevation=400):
    f = numpy.zeros([6, 4, 3])
    fh = open(fpt + ".xy", "w")
    fh.close()
    fh = open(fpt + ".xz", "w")
    fh.close()
    fh = open(fpt + ".zy", "w")
    fh.close()

    model = forward.model_dict(model)

    for i in range(problem_setup["nplt"]):
        f[:, :, :] = geom_lib.get_plate_faces_from_orientation(
            model["peast"][i],
            model["pnorth"][i],
            surface_elevation - model["ptop"][i],
            model["plngth1"][i],
            model["plngth2"][i],
            model["pwdth1"][i],
            model["pwdth2"][i],
            problem_setup["pthk"][i],
            numpy.deg2rad(model["pdzm"][i]),
            numpy.deg2rad(model["pdip"][i]),
            numpy.deg2rad(problem_setup["plng"][i]),
        )

        fd = {}
        for i in range(6):
            fd[i] = numpy.mean(f[i, :, 2])
        fds = sorted(fd.items(), key=operator.itemgetter(1), reverse=False)
        fh = open(fpt + ".xy", "a")
        for tlp in fds:
            i = tlp[0]
            fh.write(">\n")
            for j in range(4):
                fh.write("{} {}\n".format(f[i, j, 0], f[i, j, 1]))
        fh.close()

        fd = {}
        for i in range(6):
            fd[i] = numpy.mean(f[i, :, 1])
        fds = sorted(fd.items(), key=operator.itemgetter(1), reverse=True)
        fh = open(fpt + ".xz", "a")
        for tlp in fds:
            i = tlp[0]
            fh.write(">\n")
            for j in range(4):
                fh.write("{} {}\n".format(f[i, j, 0], f[i, j, 2]))
        fh.close()

        fd = {}
        for i in range(6):
            fd[i] = numpy.mean(f[i, :, 0])
        fds = sorted(fd.items(), key=operator.itemgetter(1), reverse=False)
        fh = open(fpt + ".zy", "a")
        for tlp in fds:
            i = tlp[0]
            fh.write(">\n")
            for j in range(4):
                fh.write("{} {}\n".format(f[i, j, 2], f[i, j, 1]))
        fh.close()


def plot_plate_face(
    full_fpth, forward, ax, cleanup=True, surface_elevation=400, **plotting_kwargs
):
    with open(full_fpth, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    for i in range(0, len(lines), 5):
        x = [float(l.split()[0]) for l in lines[i + 1 : i + 5]]
        y = [float(l.split()[1]) for l in lines[i + 1 : i + 5]]
        if i == 5:
            plotting_kwargs = {k: v for k, v in plotting_kwargs.items() if k != "label"}
            ax.plot(x, y, **plotting_kwargs)
        else:
            ax.plot(x, y, **plotting_kwargs)
    if "xy" in full_fpth:
        if forward.n_fiducials == 1:
            tx = forward.survey_setup["tx"]
            ty = forward.survey_setup["ty"]
            ax.plot(tx, ty, "o", color="orange")
        else:
            tx_min = float("inf")
            tx_max = float("-inf")
            for i in range(forward.n_fiducials):
                survey_setup = forward.survey_setup[i]
                ax.plot(
                    survey_setup["tx"],
                    survey_setup["ty"],
                    "o",
                    color="orange",
                )
                tx_min = min(tx_min, survey_setup["tx"])
                tx_max = max(tx_max, survey_setup["tx"])
            ax.set_xlim(tx_min - 10, tx_max + 10)
    elif "xz" in full_fpth:
        ax.axhline(surface_elevation, color="black", linestyle="--")
    elif "zy" in full_fpth:
        ax.axvline(surface_elevation, color="black", linestyle="--")
        ax.invert_xaxis()
    if cleanup == True:
        os.remove(full_fpth)
    elif cleanup == "all":
        for ext in [".xy", ".xz", ".zy"]:
            os.remove(full_fpth[:-3] + ext)


def plot_plate_faces(
    fpt, forward, model, ax1, ax2, ax3, surface_elevation=400, **plotting_kwargs
):
    gmt_plate_faces(fpt, forward, forward.problem_setup, model, surface_elevation)
    plot_plate_face(
        fpt + ".xy", forward, ax1, True, surface_elevation, **plotting_kwargs
    )
    plot_plate_face(
        fpt + ".zy", forward, ax2, True, surface_elevation, **plotting_kwargs
    )
    plot_plate_face(
        fpt + ".xz", forward, ax3, True, surface_elevation, **plotting_kwargs
    )
    ax1.set_xlabel("inline (m)")
    ax1.set_ylabel("crossline (m)")
    ax2.set_xlabel("elevation (m)")
    ax2.set_ylabel("crossline (m)")
    ax3.set_xlabel("inline (m)")
    ax3.set_ylabel("elevation (m)")


def plot_plate_faces_single(fpt, option, forward, model, ax, **plotting_kwargs):
    gmt_plate_faces(fpt, forward, forward.problem_setup, model)
    plot_plate_face(fpt + "." + option, forward, ax, "all", **plotting_kwargs)
    if option == "xy":
        ax.set_xlabel("inline (m)")
        ax.set_ylabel("crossline (m)")
    elif option == "zy":
        ax.set_xlabel("elevation (m)")
        ax.set_ylabel("crossline (m)")
    else:
        ax.set_xlabel("inline (m)")
        ax.set_ylabel("elevation (m)")
    if "x" in option and forward.n_fiducials > 1:
        tx_min = float("inf")
        tx_max = float("-inf")
        for i in range(forward.n_fiducials):
            tx = forward.survey_setup[i]["tx"]
            tx_min = min(tx_min, tx)
            tx_max = max(tx_max, tx)
        ax.set_xlim(tx_min - 10, tx_max + 10)
