# Approach II: Trans-dimensional Bayesian Inversion

In this section, we use CoFI to estimate lateral variations in phase velocity across Australia via reversible-jump Markov chain Monte Carlo (RJ-MCMC) sampling ([Green 1995](https://doi.org/10.1093/biomet/82.4.711)). RJ-MCMC is is a generalization of the Metropolis-Hastings algorithm allowing for trans-dimensional parameterizations. The algorithm starts from an initial model $\mathbf{m}$ and proposes a new model $\mathbf{m}'$ based on a perturbative approach. The new model is then accepted (in which case, $\mathbf{m} \gets \mathbf{m'}$) with probability
\begin{equation}
\tag{4}
\alpha(\mathbf{m'} \mid \mathbf{m}) = 
    \underbrace{\frac{p(\mathbf{d} \mid \mathbf{m'})}{p(\mathbf{d} \mid \mathbf{m})}}_{\text{Likelihood ratio}}
    \underbrace{\frac{p(\mathbf{m'})}{p(\mathbf{m})}}_{\text{Prior ratio}}
    \underbrace{\frac{q(\mathbf{m} \mid \mathbf{m'})}{q(\mathbf{m'} \mid \mathbf{m})}}_{\text{Proposal ratio}} 
    |\mathbf{J}|,
\end{equation}
where $p(a \mid b)$ denotes the conditional probability of $a$ given $b$ and it is understood that $\alpha = \min(1, \alpha)$. In the above expression, the Jacobian $\mathbf{J}$ of the transformation from $\mathbf{m}$ to $\mathbf{m}'$ accounts for the volume change in the parameter space under the proposed transformation. Through the forward operator $\mathbf{g}$, the likelihood expresses how well a model explains the data, and reads
\begin{equation}
\tag{5}
p(\mathbf{d} | \mathbf{m}) = \frac{1}{\sqrt{(2\pi)^n |\mathbf{C}_d|}} \ \exp \left\{\frac{-\Phi(\mathbf{m})}{2} \right\},
\end{equation}
where $n$ denotes the size of the data vector, $\mathbf{C}_d$ the data covariance matrix, and
\begin{equation}
\tag{6}
\Phi(\mathbf{m}) = \left[ \mathbf{g}(\mathbf{m}) - \mathbf{d} \right]^T \mathbf{C}_d^{-1} \left[ \mathbf{g}(\mathbf{m}) - \mathbf{d} \right]
\end{equation}
is the Mahalanobis distance between observations and model predictions. 

In MCMC methods, the process of proposing a new model and deciding whether to accept it is repeated many times to build a sequence of models $\mathcal{M} = \{\mathbf{m}_t\}$, where $t$ denotes the Markov chain iteration. In practice, a \textit{burn-in period} typically precedes the generation of $\mathcal{M}$ to allow convergence of the Markov chain to a stationary distribution. Once the burn-in period is completed, the subsequent iterations are used to populate $\mathcal{M}$, providing an approximation to the posterior distribution
\begin{equation}
\tag{7}
p(\mathbf{m} \mid \mathbf{d}) \propto p(\mathbf{d} \mid \mathbf{m}) p(\mathbf{m}).
\end{equation}
