from typing import List, Dict
import joblib
import os
import numpy
import operator
import matplotlib.pyplot as plt 
import pyp223
import wrapper_geom


# ------- problem setup
problem_setup = {
    "nlyr": 2,                                  # number of layers (icl. halfspace)
    "nstat": 1,                                 # numebr of fiducials/stations
    "nplt": 1,                                  # number of thin plates
    "cellw": 25,                                # cell width
    "pthk": numpy.array([1]),                   # plates thickness
    "plng": numpy.deg2rad(numpy.array([0])),    # plates plunge (orientation)
    
}


# ------- system specification
system_spec = {
    "ncmp": 2,                                  # system spec: number of components
    "cmp": 2,                                   # system spec: active components
    "ntrn": 3,                                  # system spec: number of transmitter turns
    "txarea": 531,                              # system spec: transmitter area
    "ampt": 0,                                  # system spec: amplitude type AMPS 0
}

# ------- transmitter settings
tx_min = 115
tx_max = 281
tx_interval = 15
n_transmitters = (tx_max - tx_min - 1) // tx_interval + 1
tx = numpy.arange(150, 240, 15)
transmitters_setup = {
    "tx": tx,                                                   # transmitter easting/x-position
    "ty": numpy.array([100]*n_transmitters),                    # transmitter northing/y-position
    "tz": numpy.array([50]*n_transmitters),                     # transmitter height/z-position
    "tazi": numpy.deg2rad(numpy.array([90]*n_transmitters)),    # transmitter azimuth
    "tincl": numpy.deg2rad(numpy.array([6]*n_transmitters)),    # transmitter inclination
    "rx": tx,                                                   # receiver easting/x-position
    "ry": numpy.array([100]*n_transmitters),                    # receiver northing/y-position
    "rz": numpy.array([50]*n_transmitters),                     # receiver height/z-position
    "trdx": numpy.array([0]*n_transmitters),                    # transmitter receiver separation inline
    "trdy": numpy.array([0]*n_transmitters),                    # transmitter receiver separation crossline
    "trdz": numpy.array([0]*n_transmitters),                    # transmitter receiver separation vertical
}


# ------- read survey data
def read_gates_and_waveform(file_name="../LeroiAir.cfl") -> dict:
    with open(file_name) as f:
        lines = f.readlines()
    nsx=int(lines[2].split()[1])
    nchnl=int(lines[2].split()[4])
    swx=[]
    waveform=[]
    for i in range(nsx):
        fields=lines[3+i].split()
        swx.append(float(fields[0])/1000.0)
        waveform.append(fields[1])
    topn=[]
    tcls=[]
    for i in range(nchnl):
        fields=lines[3+nsx+i].split()
        topn.append(float(fields[0])/1000.)
        tcls.append(float(fields[1])/1000.)
    swx=numpy.array(swx)
    waveform=numpy.array(waveform)
    topn=numpy.array(topn)
    tcls=numpy.array(tcls)
    return {
        "nsx": nsx, "nchnl": nchnl, "swx": swx, "waveform": waveform, 
        "topn": topn, "tcls": tcls
    }

# (topn + tcls) / 2

survey_data = read_gates_and_waveform()


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
    "pdip": numpy.array([60])
}


# ------- wrap forward operator
class ForwardWrapper:
    
    def __init__(
        self, 
        true_model: Dict[str, numpy.ndarray], 
        problem_setup: Dict[str, numpy.ndarray], 
        system_spec: Dict[str, numpy.ndarray],
        transmitters_setup: Dict[str, numpy.ndarray], 
        survey_data: Dict[str, numpy.ndarray], 
        params_to_invert: List[str] = None, 
        data_returned: List[str] = ["vertical", "inline"]
    ):
        self.true_model = true_model
        self.problem_setup = problem_setup
        self.system_spec = system_spec
        self.survey_data = survey_data
        
        self.n_transmitters = transmitters_setup["tx"].size
        if self.n_transmitters == 1:
            self.transmitters_setup = transmitters_setup
        else:
            self.transmitters_setup = []
            for i in range(self.n_transmitters):
                self.transmitters_setup.append({
                    k: numpy.array([v[i]]) for k, v in transmitters_setup.items()
                })
        
        if params_to_invert is None:
            params_to_invert = sorted(self.true_model.keys())
        else:
            for p in params_to_invert:
                if p not in self.true_model:
                    raise ValueError(f"Invalid parameter name: {p}")
        self.params_to_invert = sorted(params_to_invert)
        
        self.data_returned = []
        for d in data_returned:
            if d not in ["vertical", "inline"]:
                raise ValueError(f"Invalid data return type: {d}. Must be either 'inline' or 'vertical'")
            self.data_returned.append(d)
        if len(self.data_returned) == 2:
            self.data_returned = ["vertical", "inline"]     # make sure the order is correct
        
        self._init_param_length()
        self.leroiair = pyp223.LeroiAir()
    
    def __call__(self, model: numpy.ndarray, return_lengths: bool = False) -> numpy.ndarray:
        model_dict = self.model_dict(model)
        model_dict = {k: numpy.deg2rad(v) if k in ["pdzm", "pdip"] else v for k, v in model_dict.items()}
        pbres = model_dict["res"][-1]
        leroiair_failure_count = 0
        xmodl = numpy.zeros([self.survey_data["nchnl"]*self.system_spec["ncmp"]])
        if self.n_transmitters == 1:
            dpred = self._call_forward(
                pbres, leroiair_failure_count, xmodl, model_dict, self.transmitters_setup
            )
        else:
            # https://joblib.readthedocs.io/en/stable/parallel.html#thread-based-parallelism-vs-process-based-parallelism
            # calling compiled extension so using `prefer="threads"`
            dpred_all = joblib.Parallel(n_jobs=self.n_transmitters, prefer="threads")(
                joblib.delayed(self._call_forward)(pbres, leroiair_failure_count, xmodl, model_dict, transmitter_setup)
                for transmitter_setup in self.transmitters_setup
            )
            dpred = numpy.concatenate(dpred_all)
            if return_lengths:
                return dpred, [len(d) for d in dpred_all]
        return dpred

    
    def _call_forward(self, pbres, failure_count, xmodl, model_dict, transmitter_setup):
        dpred = self.leroiair.formod_vtem_max_data(
            pbres=pbres,
            leroiair_failure_count=failure_count,
            xmodl=xmodl,
            **model_dict,
            **transmitter_setup, 
            **self.survey_data,
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
            "thk": (nlyr-1)*nstat,
            "peast": nplt,
            "pnorth": nplt,
            "ptop": nplt,
            "pres": nplt,
            "plngth1": nplt,
            "plngth2": nplt,
            "pwdth1": nplt,
            "pwdth2": nplt,
            "pdzm": nplt,
            "pdip": nplt
        }

    def model_dict(self, model: numpy.ndarray) -> Dict[str, numpy.ndarray]:
        model_dict = dict(self.true_model)
        i = 0
        for p in self.params_to_invert:
            try:
                model_dict[p] = model[i:i+self.param_length[p]]
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
def plot_data(model, forward, label, ax1=None, ax2=None, **kwargs):
    vertical_returned = "vertical" in forward.data_returned
    inline_returned = "inline" in forward.data_returned
    if ax1 is None:
        if vertical_returned and inline_returned:
            _, (ax1, ax2) = plt.subplots(1, 2)
        else:
            _, ax1 = plt.subplots(1, 1)
    x = (forward.survey_data["topn"] + forward.survey_data["tcls"]) / 2
    if forward.n_transmitters == 1:
        data = forward(model)
        _plot_data(x, data, vertical_returned, inline_returned, ax1, ax2, label=label, **kwargs)
    else:
        data, data_lengths = forward(model, return_lengths=True)
        i = 0
        for length in data_lengths:
            current_data = data[i:i+length]
            if i == 0:
                _plot_data(x, current_data, vertical_returned, inline_returned, ax1, ax2, label=label, **kwargs)
            else:
                _plot_data(x, current_data, vertical_returned, inline_returned, ax1, ax2, **kwargs)
            i += length

def _plot_data(x, data, vertical_returned, inline_returned, ax1, ax2, **kwargs):
    if vertical_returned and inline_returned:
        data = data.reshape(-1, 2)
        vertical = data[:,0]
        inline = data[:,1]
        ax1.plot(x, vertical, **kwargs)
        ax2.plot(x, inline, **kwargs)
        ax1.set_xlabel("mid time of gates")
        ax1.set_ylabel("(pT/s)")
        ax2.set_xlabel("mid time of gates")
        ax2.set_ylabel("(pT/s)")
    else:
        ax1.plot(x, data, **kwargs)
        ax1.set_xlabel("mid time of gates")
        ax1.set_ylabel("log of (pT/s)")

def gmt_plate_faces(fpt, forward, problem_setup, model):
    f = numpy.zeros([6, 4, 3])
    fh = open(fpt+'.xy', 'w')
    fh.close()
    fh = open(fpt+'.xz', 'w')
    fh.close()
    fh = open(fpt+'.zy', 'w')
    fh.close()
    
    model = forward.model_dict(model)

    for i in range(problem_setup["nplt"]):
        f[:, :, :] = wrapper_geom.get_plate_faces_from_orientation(
                model["peast"][i], model["pnorth"][i], model["ptop"][i], model["plngth1"][i],
                model["plngth2"][i], model["pwdth1"][i], model["pwdth2"][i],
                problem_setup["pthk"][i], numpy.deg2rad(model["pdzm"][i]), numpy.deg2rad(model["pdip"][i]), 
                numpy.deg2rad(problem_setup["plng"][i]))
        
        fd = {}
        for i in range(6):
            fd[i] = numpy.mean(f[i, :, 2]) 
        fds = sorted(fd.items(), key=operator.itemgetter(1), reverse=False)
        fh = open(fpt + '.xy', 'a')
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
        fh = open(fpt + '.xz', 'a')
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
        fh = open(fpt+'.zy', 'a')
        for tlp in fds:
            i = tlp[0]
            fh.write(">\n")
            for j in range(4):
                fh.write("{} {}\n".format(f[i, j, 2], f[i, j, 1]))
        fh.close()

def plot_plate_face(full_fpth, forward, ax, cleanup=True, **plotting_kwargs):
    with open(full_fpth, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    for i in range(0, len(lines), 5):
        x = [float(l.split()[0]) for l in lines[i+1:i+5]]
        y = [float(l.split()[1]) for l in lines[i+1:i+5]]
        if i == 5:
            plotting_kwargs = {k: v for k, v in plotting_kwargs.items() if k != "label"}
            ax.plot(x, y, **plotting_kwargs)
        else:
            ax.plot(x, y, **plotting_kwargs)
    if "xy" in full_fpth:
        if forward.n_transmitters == 1:
            tx = forward.transmitters_setup["tx"]
            ty = forward.transmitters_setup["ty"]
            ax.plot(tx, ty, "o", color="orange")
        else:
            for i in range(forward.n_transmitters):
                transmitter_setup = forward.transmitters_setup[i]
                ax.plot(transmitter_setup["tx"], transmitter_setup["ty"], "o", color="orange")
    if cleanup:
        os.remove(full_fpth)

def plot_plate_faces(fpt, forward, problem_setup, model, ax1, ax2, ax3, **plotting_kwargs):
    gmt_plate_faces(fpt, forward, problem_setup, model)
    plot_plate_face(fpt+".xy", forward, ax1, **plotting_kwargs)
    plot_plate_face(fpt+".zy", forward, ax2, **plotting_kwargs)
    plot_plate_face(fpt+".xz", forward, ax3, **plotting_kwargs)
    ax1.set_xlabel("inline (m)")
    ax1.set_ylabel("crossline (m)")
    ax2.set_xlabel("elevation (m)")
    ax2.set_ylabel("crossline (m)")
    ax3.set_xlabel("inline (m)")
    ax3.set_ylabel("elevation (m)")
