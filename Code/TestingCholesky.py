import numpy as np
from scipy.linalg import cho_solve, cho_factor
from time import perf_counter


np.random.seed(0)

n = 5000
gamma_prior = np.random.uniform(size=(n, n))
other = np.random.uniform(size=(n, n // 100))
gamma_prior = 0.5 * (gamma_prior + gamma_prior.T)
gamma_prior = gamma_prior + n * np.eye(n)
start = perf_counter()
gamma_prior_inv = np.linalg.inv(gamma_prior)
gamma_prior_inv = gamma_prior_inv @ other
print(perf_counter() - start)

start = perf_counter()
A = cho_factor(gamma_prior)
# calculate the inverse more efficiently using the Cholesky decomposition
gamma_prior_inv_chol = cho_solve(A, other)
print(perf_counter() - start)
print(np.allclose(gamma_prior_inv_chol, gamma_prior_inv))
