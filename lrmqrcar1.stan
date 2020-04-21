// From Ben Goodrich https://discourse.datamethods.org/t/fully-sequential-bayesian-clinical-trial-design-for-in-hospital-treatment-of-covid-19
// Based on section 13.3.1 of http://hbiostat.org/doc/rms.pdf
functions { // can check these match rms::orm via rstan::expose_stan_functions

  /* log-likelihood contribution for one patient over time
   *
   * @param alpha vector of intercepts
   * @param beta vector of coefficients on baseline (time = 1) predictors
   * @param gamma real scalar for patient's RE at baseline
   * @param rho real scalar for the autoregressive RE process
   * @param eps vector of scaled random effects for patient over time > 1
   * @param x row_vector of baseline predictors
   * @param y integer array of state of the patient at time t with NA -> 0
   * @param k integer scalar for the number of valid outcome categories
   * @return real value of the log-likelihood contribution for the patient
   */
  real p_log_lik(vector alpha, vector beta, real gamma, real rho, vector eps,
                 row_vector x, int[] y, int k) {
    int T = size(y);
    real eps_t = gamma;
    real x_beta = x * beta;
    real eta = x_beta + gamma;
    real out = 0;
    int y_1 = y[1];
    if (y_1 == 1) out = log1m_exp(-log1p_exp(-alpha[1] - eta));
    else if (y_1 == k) out = -log1p_exp(-alpha[k - 1]  - eta);
    else if (y_1 > 0)  out = log_diff_exp(-log1p_exp(-alpha[y_1 - 1] - eta),
                                          -log1p_exp(-alpha[y_1]     - eta));
    // if y_1 == 0, it is missing and nothing is added to out
    for (t in 2:T) {
      int y_t = y[t];
			eps_t *= rho;
			eps_t += eps[t - 1];
			eta += eps_t;
      if (y_t == 1) out += log1m_exp(-log1p_exp(-alpha[1] - eta));
      else if (y_t == k) out += -log1p_exp(-alpha[k - 1]  - eta);
      else if (y_t >  0) out += log_diff_exp(-log1p_exp(-alpha[y_t - 1] - eta),
                                             -log1p_exp(-alpha[y_t]     - eta));
      // if y_t == 0, it is missing and nothing is added to out
    }
    return out;
  }
}
data {
  int<lower = 1> Nc;    // number of patients
  int<lower = 1> p;     // number of predictors
  matrix[Nc, p]  X;     // matrix of CENTERED baseline predictors WITH TREATMENT LAST
  int<lower = 2> k;     // number of outcome categories (7)
	int<lower = 2> Nt;    // number of time points (= maximum integer time)
  int<lower = 0, upper = k> y[Nc, Nt]; // outcome on 1 ... k with NA -> 0

  // prior standard deviations
  vector<lower = 0>[p] sds;
  real<lower = 0> rate;
	real<lower = 0> ratew;    // for within-subject white noise sigmaw
}

transformed data {
  matrix[Nc, p] Q_ast = qr_thin_Q(X);
  matrix[p, p]  R_ast = qr_thin_R(X);
  real corner = R_ast[p, p];
  matrix[p, p] R_ast_inverse = inverse(R_ast);
  row_vector[p] Q_list[Nc];

  // renormalize so that R_ast[p, p] = 1
  // THIS IMPLIES THE TREATMENT EFFECT IS UNAFFECTED BY THE ROTATION
  Q_ast *= corner;
  R_ast /= corner;
  R_ast_inverse *= corner;
  
  for (n in 1:Nc) Q_list[n] = Q_ast[n, ];
}

parameters {
  vector[p] theta; // coefficients on Q_ast
  simplex[k] pi;  // category probabilities for a person w/ average predictors
  vector[Nc] gamma_raw;  // unscaled random effects
  real<lower = 0> sigmag;   // SD of random effects
	real<lower = 0, upper = 1> rho;   // AR(1) correlation, presumed positive
	vector[Nt - 1] eps_raw[Nc];       // unscaled within-subject noise
	real<lower = 0> sigmaw;           // SD of within-subject noise
}

transformed parameters {
  vector[k - 1] alpha;                               // intercepts
  vector[Nc] log_lik;                                // log-likelihood pieces
  for (j in 2:k) alpha[j - 1] = logit(sum(pi[j:k])); // predictors are CENTERED
  for (n in 1:Nc) {
    log_lik[n] = p_log_lik(alpha, theta, sigmag * gamma_raw[n], 
                           rho, sigmaw * eps_raw[n], Q_list[n], y[n], k);
  }
}

model {
  target += normal_lpdf(theta | 0, sds);
  // implicit: pi ~ dirichlet(ones)
  gamma_raw ~ std_normal(); // implies: gamma ~ normal(0, sigmag)
  sigmag ~ exponential(rate);
  // implicit: rho ~ uniform(0, 1)
	for (n in 1:Nc) eps_raw[n] ~ std_normal(); // implies: eps[n] ~ normal(0, sigmaw)
	sigmaw ~ exponential(ratew);
  target += log_lik;
}

generated quantities {
  vector[p] beta = R_ast_inverse * theta;            // coefficients on X
  vector[p] OR = exp(beta);
}
