import numpy as np


np.random.seed(0)

n = 10
gamma_prior = np.random.uniform(size=(n, n))
gamma_prior = 0.5 * (gamma_prior + gamma_prior.T)
gamma_prior = gamma_prior + n * np.eye(n)
gamma_prior_inv = np.linalg.inv(gamma_prior)

L = np.linalg.cholesky(gamma_prior)
print(np.allclose(gamma_prior, L @ L.T))

# calculate the inverse more efficiently using the Cholesky decomposition
LT_A_inv = np.linalg.lstsq(L, np.eye(n), rcond=None)[0]
gamma_prior_inv_chol = np.linalg.lstsq(L.T, LT_A_inv, rcond=None)[0]
print(np.allclose(gamma_prior_inv_chol, gamma_prior_inv))
