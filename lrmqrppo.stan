// Based on lrmqr.stan by Ben Goodrich
functions {
  // pointwise log-likelihood contributions
  vector pw_log_lik(vector alpha, vector beta, matrix omega, 
	                  row_vector[] X, row_vector[] Z, int[] y) {
    int N = size(X);
    vector[N] out;
    int k = max(y); // assumes all possible categories are observed
    for (n in 1:N) {
      real eta = X[n] * beta;
      int j = y[n];
  		real cj;
  		real cj1;
			if (j == 1)        cj  = -( alpha[1] + eta );
			else if (j == 2)   cj  = alpha[1] + eta;
			else               cj  = alpha[j - 1] + eta + Z[n] * omega[ , j - 2];
			if(j > 1 && j < k) cj1 = alpha[j] + eta + Z[n] * omega[ , j - 1];

      if (j == 1 || j == k) out[n] = log_inv_logit(cj);
			else out[n] = log(1./(1. + exp(-cj)) - 1./(1. + exp(-cj1)));
//      else out[n] = log_diff_exp(-log1p_exp(- cj),
//                                 -log1p_exp(- cj1));
//			else out[n] = log(-log1p_exp(-cj) + log1p_exp(-cj1));
    }
    return out;
  }
  
  // Pr(y == j)
  matrix Pr(vector alpha, vector beta, matrix omega,
	          row_vector[] X, row_vector[] Z, int[] y) {
    int N = size(X);
    int k = max(y); // assumes all possible categories are observed
    matrix[N, k] out;

		for(n in 1:N) {
      real eta = X[n] * beta;
			for(j in 1 : k) {
			  real cj;
		    real cj1;
			  if (j == 1)        cj  = -( alpha[1] + eta );
			  else if (j == 2)   cj  = alpha[1] + eta;
			  else               cj  = alpha[j - 1] + eta + Z[n] * omega[ , j - 2];
			  if(j > 1 && j < k) cj1 = alpha[j] + eta + Z[n] * omega[ , j - 1];

				if (j == 1 || j == k) out[n, j] = log_inv_logit(cj);
				else out[n, j] = log(1./(1. + exp(-cj)) - 1./(1. + exp(-cj1)));
//				else  out[n, j] = log_diff_exp(-log1p_exp(-cj),
//                                   -log1p_exp(-cj1));
//				else out[n, j] = log(-log1p_exp(-cj) + log1p_exp(-cj1));
			}
    }
    return exp(out);
  }
}
data {
  int<lower = 1> N;   // number of observations
  int<lower = 1> p;   // number of predictors
	int<lower = 1> q;   // number of non-PO predictors in Z
  matrix[N, p] X;     // matrix of CENTERED predictors WITH TREATMENT LAST
	matrix[N, q] Z;     // matrix of CENTERED PPO predictors with TREATMENT LAST
  int<lower = 2> k;   // number of outcome categories
  int<lower = 1, upper = k> y[N]; // outcome on 1 ... k
  
  // prior standard deviations
  vector<lower = 0>[p] sds;
	vector<lower = 0>[q] sdsppo;

  real<lower = 0> conc;
}

transformed data {
  matrix[N, p] Q_ast = qr_thin_Q(X);
  matrix[p, p] R_ast = qr_thin_R(X);
  real corner = R_ast[p, p];
  matrix[p, p] R_ast_inverse = inverse(R_ast);
  row_vector[p] Q_list[N];
  matrix[N, q] Q_asto = qr_thin_Q(Z);
  matrix[q, q] R_asto = qr_thin_R(Z);
  row_vector[q] Q_listo[N];
  matrix[q, q] R_ast_inverseo = inverse(R_asto);

  // renormalize so that R_ast[p, p] = 1
  // THIS IMPLIES THE TREATMENT EFFECT IS UNAFFECTED BY THE ROTATION
  Q_ast *= corner;
  R_ast /= corner;
  R_ast_inverse *= corner;
  
  for (n in 1:N) Q_list[n] = Q_ast[n, ];

  corner = R_asto[q, q];
  // renormalize so that R_asto[q, q] = 1
  // THIS IMPLIES THE TREATMENT EFFECT IS UNAFFECTED BY THE ROTATION
  Q_asto *= corner;
  R_asto /= corner;
  R_ast_inverseo *= corner;
  
  for (n in 1:N) Q_listo[n] = Q_asto[n, ];
}

parameters {
  vector[p] theta; // coefficients on Q_ast
  matrix[q, k - 2] omega;  // coefficients on Q_asto
  simplex[k] pi;  // category probabilities for a person w/ average predictors
}

transformed parameters {
  vector[k - 1] alpha;                               // intercepts
  vector[N] log_lik;                                 // log-likelihood pieces
  for (j in 2:k) alpha[j - 1] = logit(sum(pi[j:k])); // predictors are CENTERED
  log_lik = pw_log_lik(alpha, theta, omega,
                       Q_list, Q_listo, y);
}

model {
  target += log_lik;
  target += normal_lpdf(theta | 0, sds);
  target += dirichlet_lpdf(pi | rep_vector(conc, k));
	for (j in 1:(k - 2)) target += normal_lpdf(omega[ , j] | 0, sdsppo);
  // implicit: pi ~ dirichlet(ones)
}

generated quantities {
  vector[p] beta = R_ast_inverse  * theta;        // coefficients on X
	matrix[q, k - 2] tau  = R_ast_inverseo * omega; // coefficients on Z
  vector[p] OR = exp(beta);
}