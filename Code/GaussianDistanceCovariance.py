import numpy as np
from numpy import ndarray
from scipy.spatial.distance import pdist, squareform
from numba import jit


@jit(nopython=True)
def gaussian_distance_covariance(coordinates, sigma, clength) -> ndarray:
    distances = pdist(coordinates, 'euclidean')
    distances = squareform(distances)
    return sigma ** 2 * np.exp(-(distances ** 2) / (2 * (clength ** 2)))
