import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, load_npz

from ProjectionMatrixCalculation import get_projection_matricies
from GaussianDistanceCovariance import gaussian_distance_covariance


def run_algorithm():
    PLOT_ROI = True
    CALCULATE_PROJECTION_MATRICIES = True
    TRACK_PHI_A = True  # track the target function on every picture during gradient descent
    PLOT_COVARIANCE = True
    PLOT_VARIANCE = True
    PLOT_ROI_RECONSTRUCTION = True
    PLOT_D = True

    N = 30 # pixels per edge
    n = N ** 2
    k = 10 # number of angles (or X-ray images)
    mm = 1 # number of rays per sensor
    m = 20 # number of sensors
    epsilon = 1e-10

    # define the grid
    x = np.arange(N) / N
    y = np.arange(N) / N
    X, Y = np.meshgrid(x, y)
    coordinates = np.column_stack([X.ravel(), Y.ravel()])

    # define the priors
    gamma_prior = gaussian_distance_covariance(coordinates, 1, 0.05) + epsilon * np.eye(n)
    x_prior = np.zeros(n)
    noise_mean = np.zeros(k * m)

    # define ROI
    A = (X - 0.3) ** 2 + (Y - 0.4) ** 2 < 0.25 ** 2
    # A = ~A
    if PLOT_ROI:
        plt.imshow(A, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()
    # reshape ROI to (N * N, N * N) in Fortran style to mimic matlab
    A = A.flatten(order="F")
    A = csr_array(np.diag(A))
    # A = csr_array(np.eye(n))

    # define projection matricies
    if CALCULATE_PROJECTION_MATRICIES:
        offsets = np.linspace(-0.49, 0.49, m * mm)
        angles = np.linspace(-np.pi / 2, np.pi / 2, k)
        print("Starting to calculate projection matricies")
        R_k = get_projection_matricies(offsets, angles, N, mm)
        print("Calculation finished")
    else:
        try:
            R_k: csr_array = load_npz("Code/Projection_matrix.npz")
        except:
            print("Loading projection matricies failed! Recalculation started.")
            offsets = np.linspace(-0.49, 0.49, m)
            angles = np.linspace(-np.pi / 2, np.pi / 2, k)
            R_k = get_projection_matricies(offsets, angles, N, 1)
    
    x, y = np.meshgrid(offsets, angles)
    offset_angle_pairs = np.vstack([x.ravel(), y.ravel()]).T
    
    # define initial parameters d and the limit D
    d = 0.1 * np.ones(shape=(k * m,))
    D = 1
    # make sure d satisfies the boundary condition
    # d = d * np.sqrt(np.sum(1 / (d ** 2)) / D)

    # initialise parameters for algorithm
    learning_rate = 0.0001
    iter_per_picture = 10
    rng = np.random.default_rng(0)
    number_of_pictures = 10

    for i in range(number_of_pictures):
        # calculate helper matricies that remain the same during the gradient descent
        Rk_gamma_prior_Rk_T = R_k @ gamma_prior @ R_k.T
        A_gamma_prior_Rk_T = A @ gamma_prior @ R_k.T
        Rk_gamma_prior_A_T = R_k @ gamma_prior @ A.T

        # find optimal d on this picture with gradient descent
        for l in range(iter_per_picture):
            print(l + 1, iter_per_picture)
            gamma_noise = np.diag(d ** 2) + epsilon * np.eye(k * m)
            Z_k = np.linalg.inv(Rk_gamma_prior_Rk_T + gamma_noise)

            # Calculate the gradient wrt d
            A_gamma_prior_Rk_T_Zk = A_gamma_prior_Rk_T @ Z_k
            Zk_Rk_gamma_prior_A_T = Z_k @ Rk_gamma_prior_A_T

            dtheta = np.zeros_like(d)
            for j in range(k * m):
                # dgamma_noise = np.zeros_like(gamma_noise)
                # dgamma_noise[j, j] = 2 * d[j]
                dgamma_noise = csr_array((np.array([2 * d[j]]), (np.array([j]), np.array([j]))), shape=gamma_noise.shape)
                dtheta[j] = np.trace(A_gamma_prior_Rk_T_Zk @ dgamma_noise @ Zk_Rk_gamma_prior_A_T)

            d -= learning_rate * dtheta

        if TRACK_PHI_A:
            gamma_posterior = gamma_prior - gamma_prior @ R_k.T @ Z_k @ R_k @ gamma_prior
            phi_A_d = 1 / N * np.sqrt(np.trace(A @ gamma_posterior @ A.T))
            print(f"Picture {i} - Modified A-optimality target function: {round(phi_A_d, 6)} - Dose of radiation: {round(np.sum(1 / (d ** 2)), 3)} - Maximum intensity angle: {round(offset_angle_pairs[np.argmax(d)][1] * 180 / np.pi, 3)}")

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
        if PLOT_COVARIANCE:
            plot_covariance(x_prior, gamma_prior, rng, N)
        if PLOT_VARIANCE:
            plot_variance(gamma_prior, N)
        if PLOT_D:
            plot_d(d)
        if PLOT_COVARIANCE or PLOT_VARIANCE or PLOT_ROI_RECONSTRUCTION or PLOT_D:
            plt.show()

        # make new starting point for the next picture
        # d = d * np.sqrt(np.sum(1 / (d ** 2)) / D)
        d = 0.1 * np.ones(shape=(k * m,))

def plot_covariance(x_prior, gamma_posterior, rng, N):
    sample_x = rng.multivariate_normal(x_prior, gamma_posterior, size=(4,), method='cholesky')
    sample_x = sample_x.reshape(4, N, N)
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        im = ax.imshow(sample_x[i], cmap='viridis', interpolation='nearest')
        fig.colorbar(im, ax=ax)
    fig.suptitle("Samples from covariance matrix")

def plot_variance(gamma_posterior, N):
    variances = np.diag(gamma_posterior)
    variances = variances.reshape(N, N)
    fig, ax = plt.subplots()
    im = ax.imshow(variances, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title("ROI reconstruction with variance")

def plot_d(d):
    fig, ax = plt.subplots()
    ax.plot(d)


run_algorithm()
