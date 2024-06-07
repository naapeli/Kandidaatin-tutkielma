import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, load_npz

from ProjectionMatrixCalculation import get_projection_matricies
from GaussianDistanceCovariance import gaussian_distance_covariance


def run_algorithm():
    PLOT_ROI = False
    CALCULATE_PROJECTION_MATRICIES = True
    TRACK_PHI_A = True  # track the target function on every picture during gradient descent

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
    D = 1
    # make sure d satisfies the boundary condition
    d = 2 * d * np.sqrt(np.sum(1 / (d ** 2)) / D)

    # initialise parameters for algorithm
    learning_rate = 0.01
    iter_per_picture = 20
    rng = np.random.default_rng(0)
    number_of_pictures = 10

    for _ in range(number_of_pictures):
        # calculate helper matricies that remain the same during the gradient descent
        Rk_gamma_prior_Rk_T = R_k @ gamma_prior @ R_k.T

        # find optimal d on this picture with gradient descent
        for i in range(iter_per_picture):
            gamma_noise = np.diag(d ** 2) + epsilon * np.eye(k * m)

            # Calculate the gradient wrt d
            Z_k = np.linalg.inv(Rk_gamma_prior_Rk_T + gamma_noise)
            dtheta = np.zeros_like(d)
            for j in range(k * m):
                dgamma_noise = np.zeros_like(gamma_noise)
                dgamma_noise[j, j] = 2 * d[j]
                dtheta[j] = np.trace(A @ gamma_prior @ R_k.T @ Z_k @ dgamma_noise @ Z_k @ R_k @ gamma_prior @ A.T)

            d -= learning_rate * dtheta

            if TRACK_PHI_A:
                gamma_posterior = gamma_prior - gamma_prior @ R_k.T @ Z_k @ R_k @ gamma_prior
                phi_A_d = 1 / N * np.sqrt(np.trace(A @ gamma_posterior @ A.T))
                print(f"{i}. - Modified A-optimality target function: {phi_A_d} - Radiation boundary satisfied: {np.sum(1 / (d ** 2)) <= D} - Dose of radiation: {np.sum(1 / (d ** 2))}")

        # calculate the optimal gamma_noise and Z_k for the current picture
        gamma_noise = np.diag(d ** 2) + epsilon * np.eye(k * m)
        Z_k = np.linalg.inv(Rk_gamma_prior_Rk_T + gamma_noise)

        # update gamma_prior after finding optimal d
        gamma_posterior = gamma_prior - gamma_prior @ R_k.T @ Z_k @ R_k @ gamma_prior
        gamma_prior = gamma_posterior

        # calculate the measurement
        # using method = "cholesky" is very important to make this run faster!
        sample_x = rng.multivariate_normal(x_prior, gamma_prior, method='cholesky')
        sample_noise = rng.multivariate_normal(noise_mean, gamma_noise, method='cholesky')
        sample_y = R_k @ sample_x + sample_noise

        # calculate the new x_prior
        x_posterior = x_prior - gamma_prior @ R_k.T @ Z_k @ (sample_y - R_k @ x_prior)
        x_prior = x_posterior

        # plot the current recreation of the image


        # make new starting point for the next picture
        d = np.random.uniform(size=(k * m,))
        d = 2 * d * np.sqrt(np.sum(1 / (d ** 2)) / D)


run_algorithm()
