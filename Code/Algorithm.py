import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, load_npz

from ProjectionMatrixCalculation import get_projection_matricies
from GaussianDistanceCovariance import gaussian_distance_covariance


def run_algorithm():
    PLOT_ROI = False
    CALCULATE_PROJECTION_MATRICIES = True

    N = 100 # pixels per edge
    n = N ** 2
    k = 10 # number of angles (or X-ray images)
    m = 20 # number of sensors

    # define the grid
    x = np.arange(N) / N
    y = np.arange(N) / N
    X, Y = np.meshgrid(x, y)
    coordinates = np.column_stack([X.ravel(), Y.ravel()])

    # define the priors
    gamma_prior = gaussian_distance_covariance(coordinates, 1, 0.1)

    # define ROI
    A = (X - 0.3) ** 2 + (Y - 0.4) ** 2 < 0.25 ** 2
    if PLOT_ROI:
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, A, cmap='viridis')
        plt.show()
    # reshape ROI to (1, N * N)
    A = A.reshape(1, -1)

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
    epsilon = 1e-10
    learning_rate = 0.01
    for i in range(iter):
        gamma_noise = np.diag(d ** 2) + epsilon * np.eye(k * m)
        Z_k = np.linalg.inv(R_k @ gamma_prior @ R_k.T + gamma_noise) # invB in matlab

        # For some reason, the calculation is slower with the cholesky decomposition than using normal inverse
        # L = np.linalg.cholesky(Z_k)
        # Z_k_inv_times_R_k_times_gamma_prior = np.linalg.lstsq(L, R_k @ gamma_prior, rcond=None)[0]
        # Z_k_inv_times_R_k_times_gamma_prior = np.linalg.lstsq(L.T, Z_k_inv_times_R_k_times_gamma_prior, rcond=None)[0]
        # gamma_posterior = gamma_prior - gamma_prior @ R_k.T @ Z_k_inv_times_R_k_times_gamma_prior
        Z_k_R_k_gamma_prior = Z_k @ R_k @ gamma_prior
        gamma_prior_R_k_T = gamma_prior @ R_k.T
        gamma_posterior = gamma_prior - gamma_prior_R_k_T @ Z_k_R_k_gamma_prior

        # calculate the value of phi_A(d) to keep track of it
        phi_A_d = np.trace(A @ gamma_posterior @ A.T)
        history_phi[i] = phi_A_d
        # could use cholesky decomposition for calculating Z_k_inv as in matlab code, but we already calculated it above

        # Calculate the gradient wrt d
        dtheta = np.zeros_like(d)
        for j in range(k * m):
            dgamma_noise = np.zeros_like(gamma_noise)
            dgamma_noise[j, j] = 2 * d[j]
            dtheta[j] = np.trace(A @ gamma_prior_R_k_T @ Z_k @ dgamma_noise @ Z_k_R_k_gamma_prior @ A.T)

        d -= learning_rate * dtheta
        print(f"{i}. - A-optimality target function: {phi_A_d} - Radiation boundary satisfied: {np.sum(1 / (d ** 2)) <= D} - Dose of radiation: {np.sum(1 / (d ** 2))}")

        gamma_prior = gamma_posterior


run_algorithm()
