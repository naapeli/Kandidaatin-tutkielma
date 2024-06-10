import numpy as np


np.random.seed(0)

TEST_FIRST_DERIVATIVE = False
TEST_SECOND_DERIVATIVE = True

l = 30
n = l ** 2
m = 3
k = 2
h = 0.01

A = np.zeros((l, l))
A[1:10, 0:20] = 1
A = A.reshape(-1)
A = np.diag(A)
R_k = np.random.normal(0, 1, size=(k * m, n))
gamma_prior = np.random.normal(0, 1, size=(n, n))
d = np.random.normal(0, 1, size=(m * k))
gamma_noise = np.diag(d ** 2) + 1e-10 * np.eye(m * k)
Z_k = np.linalg.inv(R_k @ gamma_prior @ R_k.T + gamma_noise)
gamma_posterior = gamma_prior - gamma_prior @ R_k.T @ Z_k @ R_k @ gamma_prior
def theta_A(difference=None):
    d_numeric = d + difference if difference is not None else d
    gamma_noise_numeric = np.diag(d_numeric ** 2) + 1e-10 * np.eye(m * k)
    Z_k_numeric = np.linalg.inv(R_k @ gamma_prior @ R_k.T + gamma_noise_numeric)
    gamma_posterior_numeric = gamma_prior - gamma_prior @ R_k.T @ Z_k_numeric @ R_k @ gamma_prior
    return np.trace(A @ gamma_posterior_numeric @ A.T)

if TEST_FIRST_DERIVATIVE:
    for i in range(m * k):
        # gradient based on formula
        dgamma_noise = np.zeros_like(gamma_noise)
        dgamma_noise[i, i] = 2 * d[i]
        dtheta = np.trace(A @ gamma_prior @ R_k.T @ Z_k @ dgamma_noise @ Z_k @ R_k @ gamma_prior @ A.T)

        # gradient based on numerical approximation
        difference = np.zeros_like(d)
        difference[i] = h
        dtheta_numeric = (theta_A(difference) - theta_A()) / h
        print(f"i = {i}")
        print(f"Relative error between numeric and formula: {np.abs(dtheta_numeric - dtheta) / np.abs(dtheta_numeric)}.")
        print(f"Absolute error between numeric and formula: {np.abs(dtheta_numeric - dtheta)}.")


if TEST_SECOND_DERIVATIVE:
    for i in range(m * k):
        for j in range(m * k):
            # second gradient based on formula
            dgamma_noise_i = np.zeros_like(gamma_noise)
            dgamma_noise_i[i, i] = 2 * d[i]
            if i == j:
                dgamma_noise_ii = np.zeros_like(gamma_noise)
                dgamma_noise_ii[i, i] = 2
                second_derivative = np.trace(A @ gamma_prior @ R_k.T @ (Z_k @ dgamma_noise_ii @ Z_k - 2 * Z_k @ dgamma_noise_i @ Z_k @ dgamma_noise_i @ Z_k) @ R_k @ gamma_prior @ A.T)
            else:
                dgamma_noise_j = np.zeros_like(gamma_noise)
                dgamma_noise_j[j, j] = 2 * d[j]
                second_derivative = -np.trace(A @ gamma_prior @ R_k.T @ (Z_k @ dgamma_noise_j @ Z_k @ dgamma_noise_i @ Z_k + Z_k @ dgamma_noise_i @ Z_k @ dgamma_noise_j @ Z_k) @ R_k @ gamma_prior @ A.T)

            # second gradient based on numerical approximation
            if i == j:
                difference = np.zeros_like(d)
                difference[i] = h
                second_derivative_numeric = (theta_A(difference) - 2 * theta_A() + theta_A(-difference)) / (h ** 2)
            else:
                difference_i = np.zeros_like(d)
                difference_i[i] = h
                difference_j = np.zeros_like(d)
                difference_j[j] = h

                second_derivative_numeric = (theta_A(difference_i + difference_j) + theta_A(-difference_i - difference_j) - theta_A(-difference_i + difference_j) - theta_A(difference_i - difference_j)) / (4 * h ** 2)

            print(f"(i, j) = {i, j}")
            print(f"Relative error between numeric and formula: {np.abs(second_derivative_numeric - second_derivative) / np.abs(second_derivative_numeric)}.")
            print(f"Absolute error between numeric and formula: {np.abs(second_derivative_numeric - second_derivative)}")
