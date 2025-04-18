### 1.1 Neighbourhood Algorithm <a name="na"></a>

The Neighbourhood Algorithm (NA) is described in two papers ([Sambridge, 1999a]() and [Sambridge, 1999b]()) and became popular for geophysical inverse problems.
We give a breif summary of the NA here for intuition.

It can be divided into two main phases:
1. Direct Search Phase
2. Appraisal Phase

As decribed in the papers, these phases use geometrical structures called Voronoi cells to define neighbourhoods of parameter space within which the value of the objective function is constant.
These neighbourhoods are then repeatedly sampled and refined to find the optimum position in the parameter space.

#### Direct Search Phase

This is a derivative-free optimisation aglorithm that results in an ensemble of points/samples/neighbourhoods distributed according to the objective function being solved.
More dense regions of points correspond to the more optimum areas of the objective function.

In brief, the phase algorithm progresses as follows:
1. Create an initial ensemble (of size $n_i$) of points by uniformly sampling the parameter space
2. Interate $n$ times:
   1. Rank the ensemble according to the objective function
   2. Select the best $n_r$ points in the ensemble to be resampled
   3. Resample the neighbourhoods/Voronoi cells of best $n_r$ points to obtain $n_s$ new samples
   4. Add the new samples to the ensemble

In the original papers the resampling is performed using a Gibbs sampler, although in theory any sampling algorithm can be used within each neighbourhood - the trick is in computing the neighbourhoods.

The result of this phase is a large ensemble of size $N = n_i + n \times n_s$ and their associated objective values, effectivly a discretisation of the objective function.

#### Appraisal Phase

This phase is used to reffine an ensemble of points, and produce new samples that are distributed according to the posterior distribution function.
Effectively, this is just a Bayesian sampling of a step-wise constant posterior.
In theory any sampling approach can be used to create the ensemble to be refined, and any sampling approach can be used to do the refining.
In the original papers the initial ensemble is created using the Direct Search Phase, and the refining is done with a Gibbs sampler.

A key advantage of this phase is that it avoids computing the forward function for all the new samples.
So if you have already partially sampled the parameter space but the forward function is too slow that your initial method will take too long to get a sufficient number of samples, you can use the appraisal phase of the NA.

