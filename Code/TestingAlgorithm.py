import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, load_npz, save_npz

from ProjectionMatrixCalculation import get_projection_matricies
from GaussianDistanceCovariance import gaussian_distance_covariance


def run_algorithm():
    PLOT_ROI = True
    CALCULATE_PROJECTION_MATRICIES = True # calculate the x ray matrix again or read from memory
    TRACK_PHI_A = True # track the target function on every picture during gradient descent
    PLOT_COVARIANCE = False # Take 4 samples from the posterior covariance matrix
    PLOT_STD = True # Reconstruct the image based on the standard deviations
    PLOT_D = True # plot the vector d as a function of it's indicies
    PLOT_D_IN_SAME_PICTURE = False

    N = 25 # pixels per edge
    n = N ** 2
    k = 4 # number of angles (or X-ray images)
    mm = 10 # number of rays per sensor
    m = 20 # number of sensors
    epsilon = 1e-10
    barrier_const = 0.01
    # uniform line search parameters
    max_length = 2
    n_test_points = 10

    # define the grid
    # x = np.arange(N) / N
    x = np.linspace(-0.5, 0.5, N)
    # y = np.arange(N) / N
    y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(x, y)
    coordinates = np.column_stack([X.ravel(), Y.ravel()])

    # define the priors
    gamma_prior = gaussian_distance_covariance(coordinates, 1, 0.05) + epsilon * np.eye(n)
    x_prior = np.zeros(n)
    noise_mean = np.zeros(k * m)

    # define ROI
    A = (X - 0.1) ** 2 + (Y - 0.1) ** 2 < 0.25 ** 2
    if PLOT_ROI:
        plt.imshow(A, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.show()
    # reshape ROI to (N * N, N * N) in Fortran style to mimic matlab
    A = A.flatten(order="F") # this is correct!!!
    A = csr_array(np.diag(A)) # after this A is the same as Weight in matlab
    # A = csr_array(np.eye(n))

    # define projection matricies
    if CALCULATE_PROJECTION_MATRICIES:
        offsets = np.linspace(-0.49, 0.49, m * mm)
        angles = np.linspace(-np.pi / 2, np.pi / 2 - np.pi / k, k)
        print("Starting to calculate projection matricies")
        R_k = get_projection_matricies(offsets, angles, N, mm)
        save_npz("Code/Projection_matrix.npz", R_k)
        print("Calculation finished")
    else:
        try:
            R_k: csr_array = load_npz("Code/Projection_matrix.npz")
        except:
            print("Loading projection matricies failed! Recalculation started.")
            offsets = np.linspace(-0.49, 0.49, m)
            angles = np.linspace(-np.pi / 2, np.pi / 2, k)
            R_k = get_projection_matricies(offsets, angles, N, 1)
    
    # define initial parameters d and the limit D
    d = 0.1 * np.ones(shape=(k * m,))
    D = 2200000
    # make sure d satisfies the boundary condition
    # d = d * np.sqrt(np.sum(1 / (d ** 2)) / D)

    # initialise parameters for algorithm
    learning_rate = 0.0001
    # learning_rate = 0.001
    iter_per_round = 10
    rng = np.random.default_rng(0)
    number_of_rounds = 5

    for i in range(number_of_rounds):
        # calculate helper matricies that remain the same during the gradient descent
        Rk_gamma_prior_Rk_T = R_k @ gamma_prior @ R_k.T
        A_gamma_prior_Rk_T = A @ gamma_prior @ R_k.T
        Rk_gamma_prior_A_T = R_k @ gamma_prior @ A.T

        # find optimal d on this picture with gradient descent
        for l in range(iter_per_round):
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

            # Add barrier function to the target function to discourage d to go past the dose of radiation limit (barrier function is -const * ln(D - np.sum(1 / d ** 2)))
            dbarrier = barrier_const * 2 * (d ** -3) / (D - np.sum(1 / d ** 2))
            derivative = dtheta - dbarrier
            # derivative = dtheta
            # d -= learning_rate * derivative
            
            # use uniform line search to update d
            t_uniform = np.linspace(0, max_length, n_test_points)
            min_value = np.inf
            optimal_d = d
            Z_k_optimal = Z_k
            for t in t_uniform:
                new_d = d - t * learning_rate * derivative
                if np.sum(1 / (new_d ** 2)) >= D:
                    break
                # calculate the value of the target function and barrier at new_d
                gamma_noise = np.diag(new_d ** 2) + epsilon * np.eye(k * m)
                Z_k = np.linalg.inv(Rk_gamma_prior_Rk_T + gamma_noise)
                gamma_posterior = gamma_prior - gamma_prior @ R_k.T @ Z_k @ R_k @ gamma_prior
                value = np.trace(A @ gamma_posterior @ A.T) - barrier_const * np.log(D - np.sum(1 / d ** 2))
                if value < min_value:
                    optimal_d = new_d
                    min_value = value
                    Z_k_optimal = Z_k
            # optimal_d is the new chosen point
            d = optimal_d

            if TRACK_PHI_A:
                gamma_posterior = gamma_prior - gamma_prior @ R_k.T @ Z_k_optimal @ R_k @ gamma_prior
                phi_A_d = 1 / N * np.sqrt(np.trace(A @ gamma_posterior @ A.T))
                print(f"Round {i + 1} / {number_of_rounds} - Iteration {l + 1} / {iter_per_round} - Modified A-optimality target function: {'{:.6f}'.format(phi_A_d)} - Dose of radiation: {'{:.6f}'.format(np.sum(1 / (d ** 2)))}")

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

        # plot the current recreation of the ROI and other debugging features
        if PLOT_D_IN_SAME_PICTURE and not (PLOT_COVARIANCE or PLOT_STD or PLOT_D):
            plt.plot(d, label=i+1)
        if PLOT_COVARIANCE:
            plot_covariance(x_prior, gamma_prior, rng, N)
        if PLOT_STD:
            plot_std(gamma_prior, N)
        if PLOT_D:
            plot_d(d, k, m)
        if PLOT_COVARIANCE or PLOT_STD or PLOT_D:
            plt.show()

        # make new starting point for the next picture
        # d = d * np.sqrt(np.sum(1 / (d ** 2)) / D)
        d = 0.1 * np.ones(shape=(k * m,))
    
    if PLOT_D_IN_SAME_PICTURE and not (PLOT_COVARIANCE or PLOT_STD or PLOT_D):
        vertical_line_indicies = np.arange(1, k) * m
        for x_coord in vertical_line_indicies:
            plt.axvline(x=x_coord, color='r', linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()

def plot_covariance(x_prior, gamma_posterior, rng, N):
    sample_x = rng.multivariate_normal(x_prior, gamma_posterior, size=(4,), method='cholesky')
    sample_x = sample_x.reshape(4, N, N)
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        im = ax.imshow(sample_x[i], cmap='viridis', interpolation='nearest', origin='lower')
        fig.colorbar(im, ax=ax)
    fig.suptitle("Samples from covariance matrix")

def plot_std(gamma_posterior, N):
    variances = np.sqrt(np.diag(gamma_posterior))
    variances = variances.reshape(N, N)
    fig, ax = plt.subplots()
    im = ax.imshow(variances, cmap='viridis', interpolation='nearest', vmin=0, vmax=1, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_title("ROI reconstruction with variance")

def plot_d(d, k, m):
    fig, ax = plt.subplots()
    ax.plot(d)
    vertical_line_indicies = np.arange(1, k) * m
    for x_coord in vertical_line_indicies:
        ax.axvline(x=x_coord, color='r', linestyle='--', alpha=0.3)


run_algorithm()
