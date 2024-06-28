from typing import List, Dict
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
    "tx": numpy.array([225.0]),                 # transmitter easting/x-position
    "ty": numpy.array([100.0]),                 # transmitter northing/y-position
    "tz": numpy.array([105.0]),                 # transmitter height/z-position
    "tazi": numpy.deg2rad(numpy.array([0.0])),  # transmitter azimuth
    "tincl": numpy.deg2rad(numpy.array([6.])),  # transmitter inclination
    "rx": numpy.array([225.0]),                 # receiver easting/x-position
    "ry": numpy.array([100.0]),                 # receiver northing/y-position
    "rz": numpy.array([105.0]),                 # receiver height/z-position
    "trdx": numpy.array([0.0]),                 # transmitter receiver separation inline
    "trdy": numpy.array([0.0]),                 # transmitter receiver separation crossline
    "trdz": numpy.array([0.0]),                 # transmitter receiver separation vertical
}


# ------- system specification
system_spec = {
    "ncmp": 2,                                  # system spec: number of components
    "cmp": 2,                                   # system spec: active components
    "ntrn": 3,                                  # system spec: number of transmitter turns
    "txarea": 531,                              # system spec: transmitter area
    "ampt": 0,                                  # system spec: amplitude type AMPS 0
}


# ------- read survey data
def read_gates_and_waveform(file_name="LeroiAir.cfl") -> dict:
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


# ------- model vector pack and unpack
def packmodel_vector(
    res: numpy.ndarray,         # layer resistivities, len(res) == nlyr
    thk: numpy.ndarray,         # layer thicknesses, len(thk) == (nlyr-1)*nstat
    peast: numpy.ndarray,       # plates easting location, len(peast) == nplt
    pnorth: numpy.ndarray,      # plates northing location, len(pnorth) == nplt
    ptop: numpy.ndarray,        # plates depth location, len(ptop) == nplt
    pres: numpy.ndarray,        # plate resistivities, len(pres) == nplt
    plngth1: numpy.ndarray,     # plate length 1, len(plngth1) == nplt
    plngth2: numpy.ndarray,     # plate length 2, len(plegth2) == nplt
    pwdth1: numpy.ndarray,      # plate width 1, len(plates_width1) == nplt
    pwdth2: numpy.ndarray,      # plate width 2, len(plates_width2) == nplt
    pdzm: numpy.ndarray,        # plate dip azimuth (orientation), len(pdzm) == nplt
    pdip: numpy.ndarray,        # plate dip (orientation), len(pdip) == nplt
) -> numpy.ndarray:
    return numpy.concatenate((
        res, thk, peast, pnorth, ptop, pres, plngth1, plngth2, pwdth1, pwdth2, 
        pdzm, pdip
    ))
    
def unpackmodel_vector(model: numpy.ndarray) -> dict:
    nlyr = problem_setup["nlyr"]
    nstat = problem_setup["nstat"]
    nplt = problem_setup["nplt"]
    res = model[:nlyr]
    thk = model[nlyr:nlyr+(nlyr-1)*nstat]
    peast = model[nlyr+(nlyr-1)*nstat:nlyr+(nlyr-1)*nstat+nplt]
    pnorth = model[nlyr+(nlyr-1)*nstat+nplt:nlyr+(nlyr-1)*nstat+2*nplt]
    ptop = model[nlyr+(nlyr-1)*nstat+2*nplt:nlyr+(nlyr-1)*nstat+3*nplt]
    pres = model[nlyr+(nlyr-1)*nstat+3*nplt:nlyr+(nlyr-1)*nstat+4*nplt]
    plngth1 = model[nlyr+(nlyr-1)*nstat+4*nplt:nlyr+(nlyr-1)*nstat+5*nplt]
    plngth2 = model[nlyr+(nlyr-1)*nstat+5*nplt:nlyr+(nlyr-1)*nstat+6*nplt]
    pwdth1 = model[nlyr+(nlyr-1)*nstat+6*nplt:nlyr+(nlyr-1)*nstat+7*nplt]
    pwdth2 = model[nlyr+(nlyr-1)*nstat+7*nplt:nlyr+(nlyr-1)*nstat+8*nplt]
    pdzm = model[nlyr+(nlyr-1)*nstat+8*nplt:nlyr+(nlyr-1)*nstat+9*nplt]
    pdip = model[nlyr+(nlyr-1)*nstat+9*nplt:nlyr+(nlyr-1)*nstat+10*nplt]
    return {
        "res": res, "thk": thk, "peast": peast, "pnorth": pnorth, "ptop": ptop, 
        "pres": pres, "plngth1": plngth1, "plngth2": plngth2, "pwdth1": pwdth1, 
        "pwdth2": pwdth2, "pdzm": pdzm, "pdip": pdip
    }


# ------- define true model
true_model = {
    "res": numpy.array([300, 1000]), 
    "thk": numpy.array([25]), 
    "peast": numpy.array([50]), 
    "pnorth": numpy.array([25]), 
    "ptop": numpy.array([30]), 
    "pres": numpy.array([1]), 
    "plngth1": numpy.array([100]), 
    "plngth2": numpy.array([100]), 
    "pwdth1": numpy.array([0]), 
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
        survey_data: Dict[str, numpy.ndarray], 
        params_to_invert: List[str] = None
    ):
        self.true_model = true_model
        self.problem_setup = problem_setup
        self.system_spec = system_spec
        self.survey_data = survey_data
        if params_to_invert is None:
            params_to_invert = sorted(self.true_model.keys())
        else:
            for p in params_to_invert:
                if p not in self.true_model:
                    raise ValueError(f"Invalid parameter name: {p}")
        self.params_to_invert = sorted(params_to_invert)
        self._init_param_length()
        self.leroiair = pyp223.LeroiAir()
    
    def __call__(self, model: numpy.ndarray) -> numpy.ndarray:
        model_dict = self.model_dict(model)
        model_dict = {k: numpy.deg2rad(v) if k in ["pdzm", "pdip"] else v for k, v in model_dict.items()}
        pbres = model_dict["res"][-1]
        leroiair_failure_count = 0
        xmodl = numpy.zeros([self.survey_data["nchnl"]*self.system_spec["ncmp"]])
        forward_params = {
            "pbres": pbres,
            "leroiair_failure_count": leroiair_failure_count,
            "xmodl": xmodl,
            **model_dict,
            **self.survey_data,
            **self.problem_setup,
            **self.system_spec
        }
        dpred = self.leroiair.formod_vtem_max_data(**forward_params)
        return dpred.reshape(-1)

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
        while i < model.size:
            for p in self.params_to_invert:
                model_dict[p] = model[i:i+self.param_length[p]]
                i += self.param_length[p]
        return model_dict
    
    def model_vector(self, model: Dict[str, numpy.ndarray]) -> numpy.ndarray:
        return numpy.concatenate([model[p] for p in self.params_to_invert])


forward = ForwardWrapper(true_model, problem_setup, system_spec, survey_data)


# ------- wrap plotting functions
def plot_data(model, label, ax1=None, ax2=None, **kwargs):
    if ax1 is None:
        _, (ax1, ax2) = plt.subplots(1, 2)
    data = forward(model).reshape((2,-1))
    vertical = data[:,0]
    inline = data[:,1]
    ax1.plot(vertical, label=label, **kwargs)
    ax2.plot(inline, label=label, **kwargs)


def gmt_plate_faces(fpt, problem_setup, model):
    f = numpy.zeros([6, 4, 3])
    fh = open(fpt+'.xy', 'w')
    fh.close()
    fh = open(fpt+'.xz', 'w')
    fh.close()
    fh = open(fpt+'.zy', 'w')
    fh.close()

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

def plot_plate_face(full_fpth, ax, cleanup=True, **plotting_kwargs):
    with open(full_fpth, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    for i in range(0, len(lines), 5):
        x = [float(l.split()[0]) for l in lines[i+1:i+5]]
        y = [float(l.split()[1]) for l in lines[i+1:i+5]]
        ax.plot(x, y, "-", **plotting_kwargs)
    if cleanup:
        os.remove(full_fpth)

def plot_plate_faces(fpt, problem_setup, model, ax1, ax2, ax3, **plotting_kwargs):
    gmt_plate_faces(fpt, problem_setup, model)
    plot_plate_face(fpt+".xy", ax1, **plotting_kwargs)
    plot_plate_face(fpt+".zy", ax2, **plotting_kwargs)
    plot_plate_face(fpt+".xz", ax3, **plotting_kwargs)
    ax1.set_xlabel("inline (m)")
    ax1.set_ylabel("crossline (m)")
    ax2.set_xlabel("elevation (m)")
    ax2.set_ylabel("crossline (m)")
    ax3.set_xlabel("inline (m)")
    ax3.set_ylabel("elevation (m)")
