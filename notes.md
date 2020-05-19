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
* FH question:  Since in the limit with no ties at all the first intercept when betas are all zero or covariates are all at their mean perhaps, is first intercept is logit((n-1)/n) and the last is login(1/n), would U(logit(n-1)/n), logit(1/n)) be appropriate?  This is for the model form P(Y >= y).  One complication in that way of thinking is if there are clustered observations the probability would perhaps not use n but would use the number of clusters (?).
* I previously attempted to use some uniform priors for the intercepts without much success.  I think there are a few problems with using U(logit((n-1)/n), logit(1/n)). First these are not necessarily bounds for the posterior *distributions* of the first and last intercept; these are values of the MLEs. Also, if you use an independent U(logit((n-1)/n), logit(1/n)) prior for each intercept, there is nothing in the form of the prior to enforce the ordering constraint. We can put an order constraint on using Stan but this is pretty inefficient (this is the same reason why using brms for ordinal regression is inefficient with many categories).  Both of these lead to sampling issues since we have so little data to inform the location of the intercepts in the continuous case.   Last, if we're truly in a continuous data setting, we don't know the value of n before collecting the data, so we wouldn't be able define the limits of the uniform distribution. To me, using the observed data to define the prior (even in this somewhat limited way) is a red flag that we're "double-dipping". We should be able to define the prior without seeing any data. That's what I'm trying to do with the Dirichlet process prior.
* Ben Goodrich reply: Dirichlet process makes sense conceptually, although you can't do it literally in Stan because it concentrates on a finite set. Some people have had some success with a giant but finite number of components in a Dirichlet. Also, this reminds me of some stuff that other people were doing to use a spline or similar function for the baseline hazard in survival models, which also has to be strictly increasing.
* Nathan: Ben - I agree with you about implementing a 'true' Dirichlet process in Stan; I think it will be difficult using HMC even with a finite Dirichlet process so I may have to venture outside the Stan-verse. As you mentioned there are close ties to other semi-parametric models such as those used in survival analysis.  Frank - Can you elaborate more on the conditioning on sample size? In many models we assume n is fixed rather than a parameter, but the value of n doesn't play any part in defining the form or number of parameters. Shouldn't we be able to make draws from the prior distribution without knowing the sample size? It seems that assuming knowledge about the number of outcome categories for a traditional ordinal regression is qualitatively different than assuming we know the number of distinct continuous outcome values, but I'm not sure I can explain why.
* Frank: The comment about a prior needing to have a kind of universal meaning is very interesting.  The prior should capture the pre-data state of knowledge, and if we have knowledge that uses n (which we're already conditioning on) but does not use the data in any other way, perhaps this is OK.

## Existing `Stan` Approaches

* See [this](ttps://mc-stan.org/docs/2_23/stan-users-guide/ordered-logistic-section.html)
* Nathan: The model they show is the Stan manual uses improper priors for the cutpoints with only the 'ordered' constraint in the parameter block to make sure they are increasing. The order constraint itself converts to an unconstrained space using a log difference transformation, i.e. delta_1 = alpha_1, delta_2 = log(alpha_2 - alpha_1), delta_3 = log(alpha_3 - alpha_2). If the prior on the constrained space is improper, then the unconstrained prior will also be improper so a proper posterior distribution is not guaranteed.  Usually, if there is a reasonable amount of observed data for each category then you can get convergence, but if the amount of data for each category is low (e.g. continuous CPM case or traditional ordinal model with low or zero counts in one or more categories) there can be problems getting the model to converge.
* Michael Betancourt has a good blog post that goes into more detail and also describes the alternative Dirichlet prior [here](https://betanalpha.github.io/assets/case_studies/ordinal_regression.html)

# QR Decomposition for Orthonormalization

From Ben Goodrich

The last $\beta$ parameter is unchanged so its $\beta$ equals its $\theta$ parameter in the Stan program. This will sometimes create a warning message (e.g. when using `pairs()` that "beta[...] is duplicative").

However, compiling an entire Stan program just to obtain a rescaled QR decomposition is overkill because the same can be accomplished with just R code. So, if you replace your stanQr() function with this function that does what your qr.stan program does:

```
stanQr <- function(X, center = TRUE) {
  p <- ncol(X)
  N <- nrow(X)
  if (center) X <- scale(X, center = TRUE, scale = FALSE)
  QR <- qr(X)
  Q <- qr.Q(QR)
  R <- qr.R(QR)
  sgns <- sign(diag(R))
  Q_ast <- sweep(Q, MARGIN = 2, STATS = sgns, FUN = `*`)
  R_ast <- sweep(R, MARGIN = 1, STATS = sgns, FUN = `*`)
  corner <- R_ast[p, p]
  R_ast_inverse <- backsolve(R_ast, diag(p))
  Q_ast <- Q_ast * corner
  R_ast <- R_ast / corner
  R_ast_inverse <- R_ast_inverse * corner
  return(list(Xqr = Q_ast, R = R_ast, R_inv = R_ast_inverse))
}
```

Then you also get the necessary equivalence. Moreover, the bottom right element of w$R and w$R_inv is 1.0 so the last coefficient is the same even though the design matrices are different.

All of the columns of X are different from all of the columns of Q_ast (which is what you are calling Xqr). They have to be; otherwise they wouldn't be orthogonal. But the sequence of equalities still holds

eta = X * beta = (Q * R) * beta = (Q * corner * 1 / corner * R) * beta = (Q_ast * R_ast) * beta = Q_ast * (R_ast * beta) = Q_ast * theta

where theta = R_ast * beta. So, if you do the model in terms of Q_ast and get (the posterior distribution of) the K-vector theta. When you pre-multiply theta by the upper-triangular matrix R_ast^{-1}, since the lower-right corner of both R_ast and R_ast^{-1} is 1.0, beta[K] = 0 * theta[1] + 0 * theta[2] + ... + 0 * theta[K - 1] + 1.0 * theta[K] = theta[K]. In other words, although you changed the design matrix, you left the K-th coefficient as is and changed the preceding K - 1 coefficients.

In the case of a continuous outcome and no link function, the same idea is used to get least-squares estimates

https://en.wikipedia.org/wiki/Numerical_methods_for_linear_least_squares#Orthogonal_decomposition_methods

where theta = Q' * y and then you pre-multiply theta by R^{-1}. We just added the renormalization to make the lower-right element 1.0 so that it is easy to specify a prior on theta[K].

Q: Does this guarantee that the prior for the last beta does really get applied to the treatment effect and only the treatment effect?

A: Yes, if the priors on the elements of theta are independent, which is the way we have written it in the .stan programs and would seem to be the only sensible choice since the columns of Q_ast are orthogonal to each other.

Q: If I were to split the design matrices into two matrices (and for this case the 2nd matrix has one column), apply QR to the first and not the second, and combine into one log likelihood evaluation, would I get exactly the same results as the "corner QR" method?

Generally, no. In the example you gave earlier, if you split the third column of X out, then it remains highly correlated with the first two:

```
w <- stanQr(X[,1:2])
zapsmall(cor(cbind(w$Xqr, scale(X[,3], center = TRUE, scale = FALSE))))
          [,1]      [,2]      [,3]
[1,] 1.0000000 0.0000000 0.5685352
[2,] 0.0000000 1.0000000 0.5932746
[3,] 0.5685352 0.5932746 1.0000000
```

In the case where the last variable is randomized, then the correlation would be close to zero, but it isn't exactly the same as orthogonalizing.




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

Ben Goodrich re warnings about Pareto k diagnostics from `loo()`: The `loo()` function is trying to estimate what would happen if 1 patient were dropped from the analysis and predicted conditional on all the other patients. But its importance sampling has infinite variance when the Pareto k for the left-out observation is greater than 1 and has too high variance if the Pareto k is greater than 0.7. The Pareto k pertains to how much the posterior distribution would change if one observation were left out. In these models, if one person were left out the corresponding gamma and that patient's column of eps_raw would revert to their prior distributions because there would no longer be any information in the data to update them with. Thus, this posterior distribution is too sensitive to the particular patients being conditioned on to estimate the expected log predictive density of future patients well.

To overcome this problem, we can do K-fold with K equal to the number of patients or redo the loo calculation to use a likelihood function that integrates the random effects out like in a Frequentist estimator, as described in that Psychometrica article I sent you the link to. But we can still use Stan to first obtain the posterior draws conditional on the random effects.

FH question on homescedasticity of random effects:  With random effect for subject $i$ at the first time $t=1$ having variance $\sigma^2_\gamma$, we can use the recursive relationship $V(X_t) = \rho^2 V(X_{t-1}) + \sigma^2_\epsilon$ to get variances at other times, where $\sigma^2_\epsilon$ is the within-subject white noise variance.  The variance of the random effect at time $t=2$ is $\rho^2 \sigma^2_\gamma + \sigma^2_\epsilon$ and equating the two successive variances results in $\sigma_\epsilon = \sigma_\gamma \sqrt{1 - \rho^2}$.  The same equation results from equating the variance at $t=3$ to the variance at $t=2$.   So it is reasonable to not make $\sigma_\epsilon$ a free parameter but instead to derive it from $\sigma_\gamma$ and $\rho$?  Would this make posterior sampling behave much better too?

<a name="ppo"></a>

# Partial Proportional Odds Model

The [PPO model](http://hbiostat.org/papers/feh/pet90par.pdf) of Peterson and Harrell (1990) in its *unconstrained* form (Eq. (5) of the reference) has this specification for a single observation when $Y=1, 2, ..., k$ when $j > 1$ (the paper uses a different coding, for $Y=0, ..., k$ so their $k$ is our $k-1$)):

$$P(Y \geq j | X) = \text{expit}(\alpha_{j-1} + X\beta + [j > 2] Z\tau_{j-2}) = \text{expit}(c_{j})$$

* $c_{1}$: undefined and unused
* $\alpha$: $k-1$ vector of overall intercepts
* $\tau$:  $k-2 \times q$ matrix of parameters

For the entire dataset $Z$ is an $n\times q$ design matrix specifying the form of departures from proportional odds, and to honor the hierarchy principle for interactions must be a subset of the columns of $X_{n, p}$.  With regard to $Z$ the model is multinomial instead of ordinal, and so unlike the PO model there are issues with cell sizes in the $Y$ frequency distribution.  The unconstrained PPO model is strictly for discrete ordinal $Y$ and there must be at least $k=3$ levels of $Y$.  The $\alpha$s are the intercepts for $Y \geq 2$ (and thus their negatives are intercepts for $Y=1$).

Likelihood components are as follows:

* $Y=1$: $\text{expit}(- c_2)$ ($Z$ ignored)
* $Y=2, ..., k-1$: $\text{expit}(c_Y) - \text{expit}(c_{Y+1})$ ($Z$ ignored in first term when $Y=2$)
* $Y=k$: $\text{expit}(c_k)$ ($Z$ used)

In `Stan` code `prmqrcppo.stan`, $Z$ is orthonormalized by the QR decomposition, and the PPO parameters on this new scale are $\omega$ instead of $\tau$ just as $\theta$ is substituted for $\beta$.  This code implements cluster (random) effects, so if there are no repeated observations per subject the user needs to specify a very small mean for the exponential prior for the variance of random effects (e.g., 0.001).

The $p$-vector of normal prior standard deviations for $\theta$ is `sds` and a separate $k-2 \times q$ matrix of normal prior SDs for $\omega$ is given by `sdsppo`.  If a binary treatment variable is present and one wants full control over its prior SD, be sure to put treatment as the last parameter for $X$, and if treatment is allowed to violate the PO assumption also put treatment as the last parameter for $Z$.

To some extent the constrained PPO model may be obtained by using very skeptical priors on the $\omega$ (transformed $\tau$) parameters, e.g., standard deviations < 1.0.  This will lower the effective number of model parameters.

# Pertinent Stan Documentation

* [Basics](https://mc-stan.org/docs/2_23/stan-users-guide/basic-motivation.html)
* [Multi-indexes](https://mc-stan.org/docs/2_23/stan-users-guide/multi-indexing-chapter.html)
* [Mixed operations](https://mc-stan.org/docs/2_23/functions-reference/mixed-operations.html)
* [log Sums](https://mc-stan.org/docs/2_23/stan-users-guide/log-sum-of-exponentials.html)
* [Reparameterization](https://mc-stan.org/docs/2_23/stan-users-guide/QR-reparameterization-section.html)
* [Ordered logisitic](https://mc-stan.org/docs/2_23/stan-users-guide/ordered-logistic-section.html)

## Especially Relevant Functions

| Function        | Computes |
|-----------------|----------|
| log1m_exp       | log(1 - exp(x)) |
| log1p_exp       | log(1 + exp(x)) |
| log_diff_exp    | log(exp(x) - exp(y)) |
|  -log1p_exp(-x) | log(expit(x)) |


Also study `bernoulli_logit` and `categorical_logit`.

# Other Useful Links
* [Logistic model with random intercept and small cluster sizes](https://discourse.mc-stan.org/t/bias-in-main-effects-for-logistic-model-with-random-intercept-and-small-cluster-sizes)
* [Implicit parameters as with marginal structural models](https://discourse.mc-stan.org/t/algebra-solver-has-a-side-effect-on-log-probability)
