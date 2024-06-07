import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, load_npz

from ProjectionMatrixCalculation import get_projection_matricies
from GaussianDistanceCovariance import gaussian_distance_covariance


def run_algorithm():
    PLOT_ROI = False
    CALCULATE_PROJECTION_MATRICIES = True
    MEASUREMENT = True

    N = 49 # pixels per edge
    n = N ** 2
    k = 10 # number of angles (or X-ray images)
    m = 20 # number of sensors
    epsilon = 1e-10

    # define the grid
    x = np.arange(N) / N
    y = np.arange(N) / N
    X, Y = np.meshgrid(x, y)
    coordinates = np.column_stack([X.ravel(), Y.ravel()])

    # define the priors
    gamma_prior = gaussian_distance_covariance(coordinates, 1, 0.5) + epsilon * np.eye(n)
    x_prior = np.zeros(n)
    noise_mean = np.zeros(k * m)

    # define ROI
    A = (X - 0.3) ** 2 + (Y - 0.4) ** 2 < 0.25 ** 2
    A = ~A
    if PLOT_ROI:
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, A, cmap='viridis')
        plt.show()
    # reshape ROI to (N * N, N * N) on diagonals
    A = A.flatten()
    A = np.diag(A)

    # define projection matricies
    if CALCULATE_PROJECTION_MATRICIES:
        offsets = np.linspace(-0.49, 0.49, m)
        angles = np.linspace(-np.pi / 2, np.pi / 2, k)
        print("Starting to calculate projection matricies")
        R_k = get_projection_matricies(offsets, angles, N, 1)
        print("Calculation finished")
        # print(R_k.todense())
    else:
        try:
            R_k: csr_array = load_npz("Code/Projection_matrix.npz")
        except:
            print("Loading projection matricies failed! Recalculation started.")
            offsets = np.linspace(-0.49, 0.49, m)
            angles = np.linspace(-np.pi / 2, np.pi / 2, k)
            R_k = get_projection_matricies(offsets, angles, N, 1)
    
    # define initial parameters d and the limit D
    d = np.random.uniform(size=(k * m,))
    # d = np.ones(shape=(k * m,))
    D = 1
    # make sure d satisfies the boundary condition
    d = 2 * d * np.sqrt(np.sum(1 / (d ** 2)) / D)

    iter = 100
    history_phi = np.zeros(iter)
    learning_rate = 0.01
    rng = np.random.default_rng(0)
    for i in range(iter):
        gamma_noise = np.diag(d ** 2) + epsilon * np.eye(k * m)

        # For some reason, the calculation is slower with the cholesky decomposition than using normal inverse
        # Probably because R_k @ gamma_prior is a matrix and not a vector???
        # L = np.linalg.cholesky(Z_k)
        # Z_k_inv_times_R_k_times_gamma_prior = np.linalg.lstsq(L, R_k @ gamma_prior, rcond=None)[0]
        # Z_k_inv_times_R_k_times_gamma_prior = np.linalg.lstsq(L.T, Z_k_inv_times_R_k_times_gamma_prior, rcond=None)[0]
        # gamma_posterior = gamma_prior - gamma_prior @ R_k.T @ Z_k_inv_times_R_k_times_gamma_prior

        Z_k = np.linalg.inv(R_k @ gamma_prior @ R_k.T + gamma_noise) # invB in matlab
        R_k_gamma_prior = R_k @ gamma_prior
        gamma_prior_R_k_T_Z_k = gamma_prior @ R_k.T @ Z_k
        gamma_posterior = gamma_prior - gamma_prior_R_k_T_Z_k @ R_k_gamma_prior

        # calculate the measurement
        if MEASUREMENT:
            # using method = "cholesky" is very important to make this run faster!
            sample_x = rng.multivariate_normal(x_prior, gamma_prior, method='cholesky')
            sample_noise = rng.multivariate_normal(noise_mean, gamma_noise, method='cholesky')
            sample_y = R_k @ sample_x + sample_noise

            x_posterior = x_prior - gamma_prior_R_k_T_Z_k @ (sample_y - R_k @ x_prior)
            x_prior = x_posterior

        # calculate the value of phi_A(d) to keep track of it
        phi_A_d = 1 / N * np.sqrt(np.trace(A @ gamma_posterior @ A.T))
        # modified A-optimality target
        history_phi[i] = phi_A_d
        # could use cholesky decomposition for calculating Z_k_inv as in matlab code, but we already calculated it above

        # Calculate the gradient wrt d
        dtheta = np.zeros_like(d)
        for j in range(k * m):
            dgamma_noise = np.zeros_like(gamma_noise)
            dgamma_noise[j, j] = 2 * d[j]
            dtheta[j] = np.trace(A @ gamma_prior_R_k_T_Z_k @ dgamma_noise @ Z_k @ R_k_gamma_prior @ A.T)

        d -= learning_rate * dtheta
        print(f"{i}. - Modified A-optimality target function: {phi_A_d} - Radiation boundary satisfied: {np.sum(1 / (d ** 2)) <= D} - Dose of radiation: {np.sum(1 / (d ** 2))}")

        gamma_prior = gamma_posterior


run_algorithm()
