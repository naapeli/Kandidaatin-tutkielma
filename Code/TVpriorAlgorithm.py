import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import csc_array, load_npz, save_npz
from scipy.sparse.linalg import spsolve
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from ProjectionMatrixCalculation import get_projection_matricies


def run_algorithm():
    PLOT_ROI = True
    PLOT_TARGET = True
    CALCULATE_PROJECTION_MATRICIES = True  # calculate the x ray matrix again or read from memory
    TRACK_PHI_A = True  # track the target function on every picture during gradient descent
    PLOT_COVARIANCE = True  # Take 4 samples from the posterior covariance matrix
    PLOT_STD = True  # Reconstruct the image based on the standard deviations
    PLOT_D = True  # plot the vector d as a function of it's indicies
    PLOT_RECONSTRUCTION = True  # plot the posterior mean of the distribution for the image

    N = 30  # pixels per edge
    n = N ** 2
    k = 8  # number of angles (or X-ray images)
    mm = 2  # number of rays per sensor
    m = 20  # number of sensors
    epsilon = 1e-10
    offset_range = 0.80  # the maximum and minimum offset
    # line search parameters
    max_length = 10
    barrier_const = 0.00001

    # initialise parameters for algorithm
    learning_rate = 0.01
    iter_per_round = 20
    rng = np.random.default_rng(0)
    number_of_rounds = 20

    # parameters for lagged diffusivity iteration
    tau = 1e-5
    T = 1e-6
    gamma = 10
    inv_gamma_prior = 1 / gamma ** 2 * get_H(N, np.ones(n))

    # define the grid
    x = np.linspace(-0.5, 0.5, N)
    y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(x, y)
    coordinates = np.column_stack([X.ravel(), Y.ravel()])

    # define ROI
    ROI = "whole"
    if ROI == "whole":
        A = np.ones((N, N))
    elif ROI == "offset circle":
        A = (X - 0.1) ** 2 + (Y - 0.1) ** 2 < 0.25 ** 2
    elif ROI == "bar":
        A = np.logical_and((np.abs(Y) < 0.5), X < -0.45)
    elif ROI == "left":
        A = X < 0
    elif ROI == "center circle":
        A = X ** 2 + Y ** 2 < 0.5 ** 2
    
    if PLOT_ROI:
        plt.imshow(A, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.show()
    # reshape ROI to (N * N, N * N) in Fortran style to mimic matlab
    A = A.flatten(order="F") # this is correct!!!
    A = csc_array(np.diag(A)) # after this A is the same as Weight in matlab

    # define the target
    TARGET = "ellipses"
    if TARGET == "bar":
        target_data = np.logical_and((np.abs(Y + 0.2) < 0.05), np.abs(X) < 0.45)
    elif TARGET == "circle":
        target_data = (X - 0.1) ** 2 + (Y - 0.1) ** 2 < 0.25 ** 2
    elif TARGET == "ellipses":
        amount = 3
        target_data = np.zeros_like(X)
        for _ in range(amount):
            x_loc, y_loc = np.random.uniform(-0.3, 0.3, size=2)
            a, b = np.random.uniform(0.01, 0.1, size=2)
            attenuation = np.random.uniform(0.5, 5)
            ellipse = ((X - x_loc) ** 2) / a + ((Y - y_loc) ** 2) / b < 1
            target_data[ellipse] = attenuation
    elif TARGET == "shepp-logan-phantom":
        target_data = shepp_logan_phantom()
        target_data = resize(target_data, (N, N), anti_aliasing=False)
    
    if PLOT_TARGET:
        plt.imshow(target_data, cmap='viridis', interpolation='nearest', origin='lower')
        plt.title("Target")
        plt.colorbar()
        plt.show()
    target_data = target_data.flatten(order="F")

    # define projection matricies
    if CALCULATE_PROJECTION_MATRICIES:
        offsets = np.linspace(-offset_range, offset_range, m * mm)
        angles = np.linspace(-np.pi / 2, np.pi / 2 - np.pi / k, k)
        print("Starting to calculate projection matricies")
        R_k = get_projection_matricies(offsets, angles, N, mm)
        save_npz("Code/Projection_matrix.npz", R_k)
        print("Calculation finished")
    else:
        try:
            R_k: csc_array = load_npz("Code/Projection_matrix.npz")
        except:
            print("Loading projection matricies failed! Recalculation started.")
            offsets = np.linspace(-offset_range, offset_range, m)
            angles = np.linspace(-np.pi / 2, np.pi / 2, k)
            R_k = get_projection_matricies(offsets, angles, N, 1)
    
    # define initial parameters d and the limit D
    d = 0.5 * np.ones(shape=(k * m,))
    D = 10000

    # prior covariance matrix
    gamma_prior = np.linalg.inv(inv_gamma_prior.todense())
    noise_mean = np.zeros(k * m)

    for i in range(number_of_rounds):
        # gamma_posterior from the previous iteration
        gamma_posterior = gamma_prior @ A
        gamma_prior = gamma_posterior

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

            # golden section search
            good_solution_found = False
            reduction = 1
            inv_golden_ratio = 0.618033988749895
            tolerance = 0.0001
            const = 0.5
            while not good_solution_found:
                # assume chosen points yield a feasible solution to the line search
                good_solution_found = True
                a = d
                b = d - max_length * reduction * learning_rate * derivative
                lambda_k = a + (1 - inv_golden_ratio) * (b - a)
                mu_k = a + inv_golden_ratio * (b - a)
                reduction *= const

                # make sure the parameters do not go negative
                if np.any(b < 0):
                    good_solution_found = False
                    continue

                # make sure all points are feasible, otherwise move on to a shorter interval
                if not is_valid(b, D):
                    good_solution_found = False
                    continue
                if not is_valid(lambda_k, D):
                    good_solution_found = False
                    continue
                if not is_valid(mu_k, D):
                    good_solution_found = False
                    continue

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
                            good_solution_found = False
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
                            good_solution_found = False
                            break
                        # calculate value at lambda_k
                        Z_k = get_Z_k(lambda_k, Rk_gamma_prior_Rk_T, epsilon=epsilon)
                        gamma_posterior = get_gamma_posterior(gamma_prior, R_k, Z_k)
                        lambda_k_value = phi_A(lambda_k, gamma_posterior, A, D, barrier_const=barrier_const)
            d = a if lambda_k_value < mu_k_value else b
            if not is_valid(d, D):  # should never enter here
                print("result outside of wanted region")

            if TRACK_PHI_A:
                Z_k = get_Z_k(d, Rk_gamma_prior_Rk_T, epsilon=epsilon)
                gamma_posterior = get_gamma_posterior(gamma_prior, R_k, Z_k)
                phi_A_d_modified = 1 / N * np.sqrt(phi_A(d, gamma_posterior, A, D, barrier_const=barrier_const))
                print(f"Round {i + 1} / {number_of_rounds} - Iteration {l + 1} / {iter_per_round} - Modified A-optimality target function: {'{:.6f}'.format(phi_A_d_modified)} - Dose of radiation: {'{:.6f}'.format(np.sum(1 / (d ** 2)))}")

        # do the experiment
        gamma_noise = np.diag(d ** 2) + epsilon * np.eye(k * m)
        sample_noise = rng.multivariate_normal(noise_mean, gamma_noise, method='cholesky')
        sample_y = R_k @ target_data + sample_noise

        # determine the current posterior mean and covariance matrix
        x_posterior = compute_reco(R_k, sample_y, gamma_noise, inv_gamma_prior, T, gamma, N, tau)
        x_prior = x_posterior
        edges, _ = gradient_reco(np.reshape(x_prior, (N, N), order="F"))

        # visualise edges
        plt.imshow(edges, cmap='viridis', interpolation='nearest', origin='lower')
        plt.title("Edges")
        plt.show()


        # calculate inv_gamma_prior for new optimisation round
        weight = 1 / np.sqrt(T ** 2 + edges ** 2)
        inv_gamma_prior = 1 / gamma ** 2 * get_H(N, weight)

        gamma_posterior = np.linalg.inv(inv_gamma_prior + R_k.T @ np.diag(1 / np.diag(gamma_noise)) @ R_k)
        gamma_prior = gamma_posterior

        # plot the current recreation of the ROI and other debugging features
        if PLOT_COVARIANCE:
            plot_covariance(x_prior, gamma_prior, rng, N)
        if PLOT_STD:
            plot_std(gamma_prior, N)
        if PLOT_D:
            plot_d(d, k, m)
        if PLOT_RECONSTRUCTION:
            plot_reconstruction(x_prior, N)
        if PLOT_COVARIANCE or PLOT_STD or PLOT_D or PLOT_RECONSTRUCTION:
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

def plot_reconstruction(x_prior, N):
    reconstruction = x_prior.reshape(N, N, order='F')
    fig, ax = plt.subplots()
    im = ax.imshow(reconstruction, cmap='viridis', interpolation='nearest', origin='lower')#, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title("ROI reconstruction")

def is_valid(d, D):
    return np.sum(1 / (d ** 2)) <= D

def phi_A(d, gamma_posterior, A, D, barrier_const=0.00001):
    return np.trace(A @ gamma_posterior @ A.T) - barrier_const * np.log(D - np.sum(1 / d ** 2))

def dphi_A(d, A_gamma_prior_Rk_T_Zk, Zk_Rk_gamma_prior_A_T, D, barrier_const=0.00001):
    dtheta = np.zeros_like(d)
    km = len(d)
    for j in range(km):
        # dgamma_noise = np.zeros((km, km))
        # dgamma_noise[j, j] = 2 * d[j]
        dgamma_noise = csc_array((np.array([2 * d[j]]), (np.array([j]), np.array([j]))), shape=(km, km))
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

def get_H(N, weight):
    weight = weight.reshape(N, N, order="F")
    weight1 = (np.vstack([weight, np.zeros((1, N))]) + np.vstack([np.zeros((1, N)), weight])) / 2
    weight2 = (np.hstack([weight, np.zeros((N, 1))]) + np.hstack([np.zeros((N, 1)), weight])) / 2
    sqrt_weight1 = np.sqrt(weight1.flatten(order="F"))
    sqrt_weight2 = np.sqrt(weight2.flatten(order="F"))
    Weight1 = sp.spdiags(sqrt_weight1, 0, N * (N + 1), N * (N + 1), format='csc')
    Weight2 = sp.spdiags(sqrt_weight2, 0, N * (N + 1), N * (N + 1), format='csc')

    diagonals = np.array([-np.ones(N), np.ones(N)])
    diag_places = [-1, 0]
    D = sp.spdiags(diagonals, diag_places, N + 1, N, format='csc')
    h = 1 / (N + 1)
    D /= h

    I = sp.eye(N)
    DD1 = sp.kron(I, D)
    DD2 = sp.kron(D, I)

    result1 = Weight1 @ DD1
    result2 = Weight2 @ DD2
    DD = sp.vstack([result1, result2])
    L = DD.T @ DD
    return L

def compute_reco(R, data, gamma_noise, inv_gamma_prior, T, gamma, N, tau):
    rel_diff = tau + 1
    sum = 0
    while rel_diff > tau:
        invH = lambda x: spsolve(inv_gamma_prior, x)
        B = gamma_noise + R @ invH(R.T)
        reco = invH(R.T @ np.linalg.solve(B, data))
        Reco = np.reshape(reco, (N, N), order="F")
        edges, _ = gradient_reco(Reco)
        sum_old = sum
        sum = np.sum(np.sum(edges))
        rel_diff = np.abs(sum - sum_old) / sum
        weight = 1 / np.sqrt(T ** 2 + edges ** 2)
        inv_gamma_prior = 1 / gamma ** 2 * get_H(N, weight)
    return np.reshape(Reco, N ** 2, order="F")

def gradient_reco(reco):
    N = reco.shape[0]
    diagonals = 0.5 * np.vstack([-np.ones(N), np.ones(N)])
    diag_places = [-1, 1]
    
    # Create sparse matrix D for central difference approximation
    D = sp.spdiags(diagonals, diag_places, N, N).tolil()
    
    # Correct first and last rows for forward and backward differences
    D[0, 0] = -1
    D[0, 1] = 1
    D[-1, -2] = -1
    D[-1, -1] = 1
    
    # Proper scaling
    h = 1 / (N + 1)
    D = D.tocsc() / h

    # Two dimensional derivatives as Kronecker products
    I = sp.eye(N)
    DD1 = sp.kron(I, D)
    DD2 = sp.kron(D, I)
    
    # Stack derivatives vertically
    DD = sp.vstack([DD1, DD2])

    # Computing the absolute value of the gradient
    gradientti = DD @ reco.flatten(order="F")  # Derivatives as a single vector
    reunat2 = gradientti.reshape(N**2, 2, order="F")  # Two columns correspond to two directions
    
    # Length of the gradient at each pixel
    Reunat = np.sqrt(np.sum(reunat2**2, axis=1))
    
    # Reshape back to the original size
    Reunat = Reunat.reshape(N, N, order="F")
    return Reunat, gradientti


run_algorithm()
