#!/usr/bin/env python
# coding: utf-8

# # Employ latin hypercube smapling for the objective function in three survey line example
# 
# <!-- Please leave the cell below as it is -->

# [![Open In Colab](https://img.shields.io/badge/open%20in-Colab-b5e2fa?logo=googlecolab&style=flat-square&color=ffd670)](https://colab.research.google.com/github/inlab-geo/cofi-examples/blob/main/examples/airborne_em/airborne_em_three_lines_transmitters.ipynb)

# <!-- Again, please don't touch the markdown cell above. We'll generate badge 
#      automatically from the above cell. -->
# 
# <!-- This cell describes things related to environment setup, so please add more text 
#      if something special (not listed below) is needed to run this notebook -->
# 
# > If you are running this notebook locally, make sure you've followed [steps here](https://github.com/inlab-geo/cofi-examples#run-the-examples-with-cofi-locally)
# to set up the environment. (This [environment.yml](https://github.com/inlab-geo/cofi-examples/blob/main/envs/environment.yml) file
# specifies a list of packages required to run the notebooks)

# In[5]:


# -------------------------------------------------------- #
#                                                          #
#     Uncomment below to set up environment on "colab"     #
#                                                          #
# -------------------------------------------------------- #

# !pip install -U cofi
# !pip install git+https://github.com/JuergHauser/PyP223.git


# In[6]:


# !git clone https://github.com/inlab-geo/cofi-examples.git
# %cd cofi-examples/examples/airborne_em


# In[7]:


import numpy
import smt
import smt.sampling_methods
import tqdm
from forward_lib import (
    problem_setup, 
    system_spec, 
    survey_setup, 
    ForwardWrapper
)

numpy.random.seed(42)


# # Background
# 
# This notebook performes latin hypercube smapling to creeate samples of the objective function for the three survey line example. It is recommended to covnert it into python script using the following command.
# 
# `jupyter nbconvert --to script three_survey_lines_latin_hypercube_sampling.ipynb`
# 
# This allows to run the ltain hypercube sampling as a script from the comandline to create the samples that form the training and test  dataset for the creation of a surrogate model.
# 

# In[8]:


ntrain=1000
ntest=50


# ## Problem definition

# In[9]:


tx_min = 115
tx_max = 281
tx_interval = 15
ty_min = 25
ty_max = 176
ty_interval = 75
tx_points = numpy.arange(tx_min, tx_max, tx_interval)
ty_points = numpy.arange(ty_min, ty_max, ty_interval)
n_transmitters = len(tx_points) * len(ty_points)
tx, ty = numpy.meshgrid(tx_points, ty_points)
tx = tx.flatten()
ty = ty.flatten()


# In[10]:


fiducial_id = numpy.arange(len(tx))
line_id = numpy.zeros(len(tx), dtype=int)
line_id[ty==ty_points[0]] = 0
line_id[ty==ty_points[1]] = 1
line_id[ty==ty_points[2]] = 2


# In[11]:


survey_setup = {
    "tx": tx,                                                   # transmitter easting/x-position
    "ty": ty,                                                   # transmitter northing/y-position
    "tz": numpy.array([50]*n_transmitters),                     # transmitter height/z-position
    "tazi": numpy.deg2rad(numpy.array([90]*n_transmitters)),    # transmitter azimuth
    "tincl": numpy.deg2rad(numpy.array([6]*n_transmitters)),    # transmitter inclination
    "rx": tx,                                                   # receiver easting/x-position
    "ry": numpy.array([100]*n_transmitters),                    # receiver northing/y-position
    "rz": numpy.array([50]*n_transmitters),                     # receiver height/z-position
    "trdx": numpy.array([0]*n_transmitters),                    # transmitter receiver separation inline
    "trdy": numpy.array([0]*n_transmitters),                    # transmitter receiver separation crossline
    "trdz": numpy.array([0]*n_transmitters),                    # transmitter receiver separation vertical
    "fiducial_id": fiducial_id,                                 # unique id for each transmitter
    "line_id": line_id                  # id for each line
}


# In[12]:


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
    "pdzm": numpy.array([75]),
    "pdip": numpy.array([60])
}


# In[13]:


forward = ForwardWrapper(true_model, problem_setup, system_spec, survey_setup,
                         ["pdip","pdzm", "peast", "ptop", "pwdth2"], data_returned=["vertical"])


# In[14]:


# check the order of parameters in a model vector
forward.params_to_invert


# In[15]:


true_param_value = numpy.array([60,65, 175, 30, 90])


# **Generate synthetic data**

# In[16]:


data_noise = 0.01
data_pred_true = forward(true_param_value)
data_obs = data_pred_true + numpy.random.normal(0, data_noise, data_pred_true.shape)


# ## Create surrogate model

# **Define objective function**

# In[17]:


def my_objective(model):
    dpred = forward(model)
    residual = dpred - data_obs
    return residual.T @ residual


# In[18]:


ndim=len(true_param_value)
xlimits=numpy.array([[10,90],[10,160],[150,190],[25,45],[50,150]])


# In[19]:


sampling = smt.sampling_methods.LHS(xlimits=xlimits,random_state=42)
xtrain=sampling(ntrain)
ytrain=[]
xtest=sampling(ntest)
ytest=[]


# In[20]:


for x in tqdm.tqdm(xtrain):
    ytrain.append(my_objective(x))
for x in tqdm.tqdm(xtest):
    ytest.append(my_objective(x))


# In[21]:


xtrain=numpy.array(xtrain)
ytrain=numpy.array(ytrain)

xtest=numpy.array(xtest)
ytest=numpy.array(ytest)


# In[22]:


with open('three_survey_lines_lhs.npy', 'wb') as f:
    numpy.save(f,ndim)
    numpy.save(f,xtrain)
    numpy.save(f,ytrain)
    numpy.save(f,xtest)
    numpy.save(f,ytest)


# ---
# ## Watermark
# 
# <!-- Feel free to add more modules in the watermark_list below, if more packages are used -->
# <!-- Otherwise please leave the below code cell unchanged -->

# In[24]:


watermark_list = ["cofi", "numpy", "scipy", "matplotlib","smt"]
for pkg in watermark_list:
    pkg_var = __import__(pkg)
    print(pkg, getattr(pkg_var, "__version__"))


# In[ ]:




