---
pagetitle: Stan Notes
---
# Resources
* [STAN for linear mixed models](https://people.bath.ac.uk/jjf23/stan/) by Julian Faraway, especially the penicillin example
* [Bayesian inference with Stan: A tutorial on adding custom distributions](https://link.springer.com/article/10.3758/s13428-016-0746-9) by J Annis, B Miller, T Palmeri
* [stan_lmer vs hard-coded Stan](https://github.com/kholsinger/mixed-models) 
* [RStan: the R interface to Stan](https://mc-stan.org/rstan/articles/rstan.html)
* [Multi-level ordinal regression models with brms](https://kevinstadler.github.io/blog/bayesian-ordinal-regression-with-random-effects-using-brms/)
* [Estimating generalized (non-)linear models with group-specific terms with rstanarm](https://cran.r-project.org/web/packages/rstanarm/vignettes/glmer.html)

# Inner Workings of the Bayesian Proportional Odds Model
* From Nathan James: Automatic imposition of order constraint for intercepts
   + For our Bayesian model the intuition behind the intercept ordering is that we first estimate the probabilities of being in each of the n categories (combining observed data with the Dirichlet prior) and then apply the link function to the cumulative probabilities to calculate the intercepts. Because the cumulative probabilities are ordered and the link is monotonic, the intercepts are also ordered. Here's a few lines of example R code that illustrate the process
   + estimated posterior prob. of membership in 7 categories (when X*beta=0)
   + est_probs <- c(0.11, 0.27, 0.16, 0.14, 0.04, 0.13, 0.15)
   + estimated posterior cumulative probs
   + cum_probs <- cumsum(est_probs)
   + apply logit link to cum_probs to estimate intercepts
   + est_intercepts <- qlogis(cum_probs)

# QR Decomposition for Orthonormmalization
* From Ben Goodrich: The last $\beta$ parameter is unchanged so its $\beta$ equals its $\theta$ parameter in the Stan program. This will sometimes create a warning message (e.g. when using `pairs()` that "beta[...] is duplicative").

# General Information About Random Effects
* From Jonathan Schildcrout: Omitting random effects from a model
   + This is not bias.  The two models are estimating different parameters.  For Binary data there are approximations that can be used to convert between the marginal (what LRM estimates) and conditional (what random effects estimate) parameters.  See Zeger Liang and Albert 1988.  Not sure if that works for general ordinal data. 
   + The ZLA conversion is this
   + Sigma = 2.04
   + C = 16*sqrt(3)/(15*3.14)
   + Mult = 1 / sqrt(c^2 * Sigma^2 +1) = 0.76
   + beta.marg = Mult * beta.cond
   + You can think of the sd of the random effects as a coefficient for a standardized covariate that was not observed: $Logit(p_{ij}) = x_{ij} \beta + b_i$, and $b_i ~ N(0, \sigma^2)$ which means $Logit(p_{ij}) = x_{ij} \beta + \sigma * Z_i$ , and $Z_i ~ N(0, 1)$
* Ben Goodrich: Why not use a prior that is n(0, $\sigma_{\gamma}$)
   + The main thing is that usually (but unfortunately not always) the MCMC goes better if you utilize the stochastic representation of the normal distribution, which entails putting the unscaled random effects in the parameters block and scaling them by sigmag in the transformed parameters block to get the scaled random effects that go into the likelihood function. Then, the unscaled random effects get a standard normal prior, which implies before you see the data that the scaled random effects are distributed normal with mean zero and standard deviation sigmag. That tends to reduce the posterior dependence between sigmag and the (unscaled) random effects. In addition, there should be a proper prior on sigmag. I put it as exponential, but you would need to pass in the prior rate from R.

## AR(1) Random Effects
Ben Goodrich re warnings about Pareto k diagnostics from `loo()`: The `loo()` function is trying to estimate what would happen if 1 patient were dropped from the analysis and predicted conditional on all the other patients. But its importance sampling has infinite variance when the Pareto k for the left-out observation is greater than 1 and has too high variance if the Pareto k is greater than 0.7. The Pareto k pertains to how much the posterior distribution would change if one observation were left out. In these models, if one person were left out the corresponding gamma and that patient's column of eps_raw would revert to their prior distributions because there would no longer be any information in the data to update them with. Thus, this posterior distribution is too sensitive to the particular patients being conditioned on to estimate the expected log predictive density of future patients well.

To overcome this problem, we can do K-fold with K equal to the number of patients or redo the loo calculation to use a likelihood function that integrates the random effects out like in a Frequentist estimator, as described in that Psychometrica article I sent you the link to. But we can still use Stan to first obtain the posterior draws conditional on the random effects.



# MLEs and Optimization for Random Effects Models
* From Ben Goodrich: Inconsistent results when using `rstan::optimizing`
   + Optimizing is not going to work well in hierarchical models that condition on the values of the random effects because (penalized) MLE is not a consistent estimator of (any of) the parameters as the number of clusters gets large. That is why Frequentists have to first integrate the random effects out of the likelihood before they can choose the fixed effects to maximize it. If you do MCMC, then the posterior means and medians should be much more stable than the modes. You can also try the vb() function, which implements variational Bayesian inference --- essentially trying to find the closest multivariate normal distribution to posterior distribution in the unconstrained space --- but that tends to not work well in general either.

# Prediction With Random Effects
* Ben Goodrich: Everything about the random effect stuff is easier from a Bayesian perspective, except for one thing: If you want to evaluate the model based on how it is expected to predict new patients (that by definition have not been observed yet), then you have to re-calculate the log-likelihood contributions in generated quantities after numerically integrating out the random effects like the Frequentists do. This is not so bad to code now that Stan has a one-dimensional numerical integration function, but it takes doing.  See https://arxiv.org/abs/1802.04452

# AR(1) Modeling
* <https://discourse.mc-stan.org/t/migrated-from-google-group-ar-1-logistic-regression-funnel>
* <https://discourse.mc-stan.org/t/improving-efficiency-when-modeling-autocorrelation>
* <https://discourse.mc-stan.org/t/dynamic-panel-data-models-with-stan>
* <https://github.com/jgabry/stancon2018helsinki_intro/tree/master/slides>
* <https://github.com/jgabry/stancon2018helsinki_intro/blob/master/Pest_Control_Example.Rmd>
* <https://mc-stan.org/docs/2_20/stan-users-guide/autoregressive-section.html>
* <https://www.mathworks.com/help/econ/simulate-stationary-arma-processes.html> (kick starting the process with unconditional mean)
* <https://www.math.utah.edu/~zhorvath/ar1.pdf> (recursive substitution p. 5)

The equation in the last reference p. 5 allows specification of the random effect at time t without passing through all the previous random effects, so it accounts for unequally spaced and missing time points.  It suggests this model:

* Let $\gamma_i$ be a $n(0, \sigma_\gamma$) random effect for the $i$th subject.
* Let $\epsilon_1, ... \epsilon_T$ be the within-subject white noise that is $n(0, \sigma_w)$, where $T$ is the maximum follow-up time (we may only use the first few of these for a given subject)
* Then the random effect for subject $i$ at time $t$ is $r_{i,t} = \rho^{t}\gamma_i + \rho^{t-1}\epsilon_1 + \rho^{t-2}\epsilon_2 + ... \epsilon_k$
* Or since the white noise $\epsilon$ are generated while Stan is running, they will all be defined regardless of which observations are actually observed, so the standard specification should work: $r_{i,1} = \gamma_i, r_{i,2} = \rho r_{i,1} + \epsilon_2, r_{i,3} = \rho r_{i, 2} + \epsilon_3, ...$.

Would this specification lead to sampling problems?  Do $\sigma_\gamma$ and $\sigma_w$ compete too much?




