// Return thin QR translation of a design matrix
data {
  int<lower = 1> N;   // number of observations
  int<lower = 1> p;   // number of predictors
  matrix[N, p] X;     // matrix of predictors
}

transformed data {
  matrix[N, p] Q_ast = qr_thin_Q(X);
  matrix[p, p] R_ast = qr_thin_R(X);
  real corner = R_ast[p, p];
  matrix[p, p] R_ast_inverse = inverse(R_ast);
  row_vector[p] Q_list[N];

  // renormalize so that R_ast[p, p] = 1
  // THIS IMPLIES THE TREATMENT EFFECT IS UNAFFECTED BY THE ROTATION
  Q_ast *= corner;
  R_ast /= corner;
  R_ast_inverse *= corner;
  for (n in 1:N) Q_list[n] = Q_ast[n, ];
}

generated quantities {
  row_vector[p] Xqr[N];
  matrix[p,p] Ri;
  for (n in 1:N) Xqr[n] = Q_list[n];
  Ri = R_ast_inverse;
}
