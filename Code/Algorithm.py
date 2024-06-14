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
    k = 8 # number of angles (or X-ray images)
    mm = 10 # number of rays per sensor
    m = 20 # number of sensors
    epsilon = 1e-10
    # line search parameters
    max_length = 2
    n_test_points = 10
    barrier_const = 0.00001

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
    ROI = 1
    if ROI == 0:
        A = np.ones((N, N))
    elif ROI == 1:
        A = (X - 0.1) ** 2 + (Y - 0.1) ** 2 < 0.25 ** 2
    elif ROI == 2:
        A = np.logical_and((np.abs(Y) < 0.5), X < -0.45)
    
    if PLOT_ROI:
        plt.imshow(A, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.show()
    # reshape ROI to (N * N, N * N) in Fortran style to mimic matlab
    A = A.flatten(order="F") # this is correct!!!
    A = csr_array(np.diag(A)) # after this A is the same as Weight in matlab
        


    # define projection matricies
    if CALCULATE_PROJECTION_MATRICIES:
        offsets = np.linspace(-0.8, 0.8, m * mm)
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
    d = 0.5 * np.ones(shape=(k * m,))
    D = 2200000
    # make sure d satisfies the boundary condition

    # initialise parameters for algorithm
    learning_rate = 0.01
    iter_per_round = 10
    rng = np.random.default_rng(0)
    number_of_rounds = 20

    for i in range(number_of_rounds):
        # calculate helper matricies that remain the same during the gradient descent
        Rk_gamma_prior_Rk_T = R_k @ gamma_prior @ R_k.T
        A_gamma_prior_Rk_T = A @ gamma_prior @ R_k.T
        Rk_gamma_prior_A_T = R_k @ gamma_prior @ A.T

        # find optimal d on this picture with gradient descent
        for l in range(iter_per_round):
            Z_k = get_Z_k(d, Rk_gamma_prior_Rk_T, epsilon=epsilon)

            # Calculate the gradient wrt d
            A_gamma_prior_Rk_T_Zk = A_gamma_prior_Rk_T @ Z_k
            Zk_Rk_gamma_prior_A_T = Z_k @ Rk_gamma_prior_A_T

            derivative = dphi_A(d, A_gamma_prior_Rk_T_Zk, Zk_Rk_gamma_prior_A_T, D, barrier_const=barrier_const)
            
            # # use uniform line search to update d
            # t_uniform = np.linspace(0, max_length, n_test_points)
            # min_value = np.inf
            # optimal_d = d
            # for t in t_uniform:
            #     new_d = d - t * learning_rate * derivative
            #     if not is_valid(new_d, D):
            #         break
            #     # calculate the value of the target function and barrier at new_d
            #     Z_k = get_Z_k(new_d, Rk_gamma_prior_Rk_T, epsilon=epsilon)
            #     gamma_posterior = get_gamma_posterior(gamma_prior, R_k, Z_k)
            #     value = phi_A(new_d, gamma_posterior, A, D, barrier_const=barrier_const)
            #     if value < min_value:
            #         optimal_d = new_d
            #         min_value = value
            # # optimal_d is the new chosen point
            # d = optimal_d

            # golden section search
            inv_golden_ratio = 0.618033988749895
            tolerance = 0.0001
            a = d
            # make sure to not go over the unfeasible set
            const = 0.9
            b = np.abs(d - max_length * learning_rate * derivative)
            ind = 1
            while True:
                if is_valid(b, D):
                    break
                elif ind > 1000:
                    b = d
                    break
                b = np.abs(d - (const ** ind) * max_length * learning_rate * derivative)
                ind += 1

            lambda_k = a + (1 - inv_golden_ratio) * (b - a)
            mu_k = a + inv_golden_ratio * (b - a)
            # values at the start and end points
            Z_k = get_Z_k(lambda_k, Rk_gamma_prior_Rk_T, epsilon=epsilon)
            gamma_posterior = get_gamma_posterior(gamma_prior, R_k, Z_k)
            lambda_k_value = phi_A(lambda_k, gamma_posterior, A, D, barrier_const=barrier_const)
            Z_k = get_Z_k(mu_k, Rk_gamma_prior_Rk_T, epsilon=epsilon)
            gamma_posterior = get_gamma_posterior(gamma_prior, R_k, Z_k)
            mu_k_value = phi_A(mu_k, gamma_posterior, A, D, barrier_const=barrier_const)

            while np.any(np.abs(b - a) > tolerance):
                if lambda_k_value > mu_k_value:
                    a = lambda_k
                    lambda_k = mu_k
                    mu_k = a + inv_golden_ratio * (b - a)
                    if not is_valid(mu_k, D):
                        print("upperbound outside of wanted region", np.sum(1 / mu_k ** 2), D)
                        break
                    # calculate value at mu_k
                    Z_k = get_Z_k(mu_k, Rk_gamma_prior_Rk_T, epsilon=epsilon)
                    gamma_posterior = get_gamma_posterior(gamma_prior, R_k, Z_k)
                    mu_k_value = phi_A(mu_k, gamma_posterior, A, D, barrier_const=barrier_const)
                else:
                    b = mu_k
                    mu_k = lambda_k
                    lambda_k = a + (1 - inv_golden_ratio) * (b - a)
                    if not is_valid(lambda_k, D):
                        print("lowerbound outside of wanted region", np.sum(1 / lambda_k ** 2), D)
                        break
                    # calculate value at lambda_k
                    Z_k = get_Z_k(lambda_k, Rk_gamma_prior_Rk_T, epsilon=epsilon)
                    gamma_posterior = get_gamma_posterior(gamma_prior, R_k, Z_k)
                    lambda_k_value = phi_A(lambda_k, gamma_posterior, A, D, barrier_const=barrier_const)
            d = (a + b) / 2
            if not is_valid(d, D):
                print("result outside of wanted region")

            if TRACK_PHI_A:
                Z_k = get_Z_k(d, Rk_gamma_prior_Rk_T, epsilon=epsilon)
                gamma_posterior = get_gamma_posterior(gamma_prior, R_k, Z_k)
                phi_A_d_modified = 1 / N * np.sqrt(phi_A(d, gamma_posterior, A, D, barrier_const=barrier_const))
                print(f"Round {i + 1} / {number_of_rounds} - Iteration {l + 1} / {iter_per_round} - Modified A-optimality target function: {'{:.6f}'.format(phi_A_d_modified)} - Dose of radiation: {'{:.6f}'.format(np.sum(1 / (d ** 2)))}")

        # update gamma_prior after finding optimal d
        Z_k = get_Z_k(d, Rk_gamma_prior_Rk_T, epsilon=epsilon)
        gamma_prior = get_gamma_posterior(gamma_prior, R_k, Z_k)

        # calculate the measurement
        # using method = "cholesky" is very important to make this run faster!
        gamma_noise = np.diag(d ** 2) + epsilon * np.eye(k * m)
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
    
    if PLOT_D_IN_SAME_PICTURE and not (PLOT_COVARIANCE or PLOT_STD or PLOT_D):
        vertical_line_indicies = np.arange(1, k) * m
        for x_coord in vertical_line_indicies:
            plt.axvline(x=x_coord, color='r', linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()

def plot_covariance(x_prior, gamma_posterior, rng, N):
    sample_x = rng.multivariate_normal(x_prior, gamma_posterior, size=(4,), method='cholesky')
    sample_x = sample_x.reshape(4, N, N, order='F')
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        im = ax.imshow(sample_x[i], cmap='viridis', interpolation='nearest', origin='lower')
        fig.colorbar(im, ax=ax)
    fig.suptitle("Samples from covariance matrix")

def plot_std(gamma_posterior, N):
    variances = np.sqrt(np.diag(gamma_posterior))
    variances = variances.reshape(N, N, order='F')
    fig, ax = plt.subplots()
    im = ax.imshow(variances, cmap='viridis', interpolation='nearest', origin='lower')#, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title("ROI reconstruction with variance")

def plot_d(d, k, m):
    fig, ax = plt.subplots()
    ax.plot(d)
    vertical_line_indicies = np.arange(1, k) * m
    for x_coord in vertical_line_indicies:
        ax.axvline(x=x_coord, color='r', linestyle='--', alpha=0.3)

def is_valid(d, D):
    return np.sum(1 / (d ** 2)) < D

def phi_A(d, gamma_posterior, A, D, barrier_const=0.00001):
    return np.trace(A @ gamma_posterior @ A.T) - barrier_const * np.log(D - np.sum(1 / d ** 2))

def dphi_A(d, A_gamma_prior_Rk_T_Zk, Zk_Rk_gamma_prior_A_T, D, barrier_const=0.00001):
    dtheta = np.zeros_like(d)
    km = len(d)
    for j in range(km):
        # dgamma_noise = np.zeros_like(gamma_noise)
        # dgamma_noise[j, j] = 2 * d[j]
        dgamma_noise = csr_array((np.array([2 * d[j]]), (np.array([j]), np.array([j]))), shape=(km, km))
        dtheta[j] = np.trace(A_gamma_prior_Rk_T_Zk @ dgamma_noise @ Zk_Rk_gamma_prior_A_T)

    # Add barrier function to the target function to discourage d to go past the dose of radiation limit (barrier function is -const * ln(D - np.sum(1 / d ** 2)))
    dbarrier = barrier_const * 2 * (d ** -3) / (D - np.sum(1 / d ** 2))
    derivative = dtheta - dbarrier
    return derivative

def get_gamma_posterior(gamma_prior, R_k, Z_k):
    gamma_posterior = gamma_prior - gamma_prior @ R_k.T @ Z_k @ R_k @ gamma_prior
    return gamma_posterior

def get_Z_k(d, Rk_gamma_prior_Rk_T, epsilon=1e-10):
    gamma_noise = np.diag(d ** 2) + epsilon * np.eye(len(d))
    Z_k = np.linalg.inv(Rk_gamma_prior_Rk_T + gamma_noise)
    return Z_k


run_algorithm()
