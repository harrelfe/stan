// From Ben Goodrich https://discourse.datamethods.org/t/fully-sequential-bayesian-clinical-trial-design-for-in-hospital-treatment-of-covid-19
// Based on section 13.3.1 of http://hbiostat.org/doc/rms.pdf
functions { // can check these match rms::orm via rstan::expose_stan_functions
  // pointwise log-likelihood contributions
  vector pw_log_lik(vector alpha, vector beta, row_vector[] X, int[] y, 
                    int[] cluster, vector gamma) {
    int N = size(X);
    vector[N] out;
    int k = max(y); // assumes all possible categories are observed
    for (n in 1:N) {
      real eta = X[n] * beta + gamma[cluster[n]];
      int y_n = y[n];
      if (y_n == 1) out[n] = log1m_exp(-log1p_exp(-alpha[1] - eta));
      else if (y_n == k) out[n] = -log1p_exp(-alpha[k - 1]  - eta);
      else out[n] = log_diff_exp(-log1p_exp(-alpha[y_n - 1] - eta),
                                 -log1p_exp(-alpha[y_n]     - eta));
    }
    return out;
  }
  
  // Pr(y == j)
  matrix Pr(vector alpha, vector beta, row_vector[] X, int[] y, 
            int[] cluster, vector gamma) {
    int N = size(X);
    int k = max(y); // assumes all possible categories are observed
    matrix[N, k] out;
    for (n in 1:N) {
      real eta = X[n] * beta + gamma[cluster[n]];
      out[n, 1] = log1m_exp(-log1p_exp(-alpha[1] - eta));
      out[n, k] = -log1p_exp(-alpha[k - 1]  - eta);
      for (y_n in 2:(k - 1))
        out[n, y_n] = log_diff_exp(-log1p_exp(-alpha[y_n - 1] - eta),
                                   -log1p_exp(-alpha[y_n]     - eta));
    }
    return exp(out);
  }
}
data {
  int<lower = 1> N;   // number of observations
  int<lower = 1> Nc;  // number of clusters
  int<lower = 1> p;   // number of predictors
  matrix[N, p] X;     // matrix of CENTERED predictors
  int<lower = 2> k;   // number of outcome categories (7)
  int<lower = 1, upper = k> y[N]; // outcome on 1 ... k
  int<lower = 1, upper = Nc> cluster[N];  // cluster IDs
  
  // prior standard deviations
  vector<lower = 0>[p] sds;

  int<lower = 1, upper = 2> psigma;  // 1=t(4, rsdmean, rsdsd); 2=exponential
  real<lower = 0> rsdmean;  // mean of prior for sigma
  real<lower = 0> rsdsd;    // scale parameter for sigma (used only if psigma=1)

  real<lower = 0> conc;     // for PO and PPO models
}

transformed data {
  row_vector[p] Xr[N];
  for (n in 1:N) Xr[n] = X[n, ];
}

parameters {
  vector[p] beta; // coefficients on X
  simplex[k] pi;  // category probabilities for a person w/ average predictors
  vector[Nc] gamma_raw;  // unscaled random effects
  real<lower = 0> sigmag;   // SD of random effects
}

transformed parameters {
  vector[k - 1] alpha;                               // intercepts
  vector[Nc] gamma = sigmag * gamma_raw;             // scaled random effects
  vector[N] log_lik;                                 // log-likelihood pieces
  for (j in 2:k) alpha[j - 1] = logit(sum(pi[j:k])); // predictors are CENTERED
  log_lik = pw_log_lik(alpha, beta, Xr, y, cluster, gamma);
}

model {
  gamma_raw ~ std_normal(); // implies: gamma ~ normal(0, sigmag)
  if(psigma == 1) sigmag ~ student_t(4, rsdmean, rsdsd);
  else sigmag ~ exponential(1. / rsdmean); 
  target += log_lik;
  target += dirichlet_lpdf(pi | rep_vector(conc, k));
  target += normal_lpdf(beta | 0, sds);
}
