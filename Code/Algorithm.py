import numpy as np
from scipy.sparse import lil_array
from time import perf_counter


def get_projection_matricies(offsets, angles, N, m):
    assert len(offsets) % m == 0, "The number of offsets len(offsets) must be divisible by the number of rays (m)."
    print(len(offsets), len(angles))
    x, y = np.meshgrid(offsets, angles)
    offset_angle_pairs = np.vstack([x.ravel(), y.ravel()]).T

    epsilon = 1e-10
    p = np.linspace(0, 1, N) # x or y coordinates of gridlines

    Temp = lil_array(np.zeros(shape=(N * N, m)))
    A = np.zeros(shape=(len(offset_angle_pairs) // m, N * N))
    for k, (offset, angle) in enumerate(offset_angle_pairs):
        intersection_points = np.array([[0, 0, 0]]) # t, x, y
        cos = np.cos(angle)
        sin = np.sin(angle)

        # compute intersections with vertical lines
        if np.abs(cos) > epsilon: # avoid division by zero errors
            t = 1 / cos * (p - 0.5 - sin * offset)
            x = 0.5 + offset*cos - sin * t
            aux = np.where((x >= 0) & (x <= 1))[0]
            intersection_points = np.vstack([intersection_points, np.array([t[aux], x[aux], p[aux]]).T])
        
        if np.abs(sin) > epsilon:
            t = 1 / sin * (0.5 + offset * cos - p)
            y = 0.5 + offset * sin  + cos * t
            aux = np.where((y >= 0) & (y <= 1))[0]
            intersection_points = np.vstack([intersection_points, np.array([t[aux], p[aux], y[aux]]).T])

        intersection_points = intersection_points[1:]
        # sort intersection points wrt. t
        sorted_indices = np.argsort(intersection_points[:, 0])
        intersection_points = intersection_points[sorted_indices]

        # calculate the length of intersection lines
        lengths = intersection_points[1:, 0] - intersection_points[:-1, 0]
        xmids = 0.5 * (intersection_points[1:, 1] + intersection_points[:-1, 1])
        ymids = 0.5 * (intersection_points[1:, 2] + intersection_points[:-1, 2])
        iaux = np.where(lengths > epsilon)[0]
        lengths = lengths[iaux]
        xmids = xmids[iaux]
        ymids = ymids[iaux]
        indx = np.floor(N * xmids)
        indy = np.floor(N * (1 - ymids))

        # array slicing and updating Temp takes the most amount of time
        Temp[(indx - 1) * N + indy, k % (m - 1)] = lengths

        if k % m == 0:
            kk = k // m
            A[kk] = Temp.sum(axis=1) / m
            Temp = lil_array(np.zeros(shape=(N * N, m)))
        if k % 5000 == 0:
            print(k, len(offset_angle_pairs))
    return A

m = 100
mm = 133
N = 49
n = 21
nn = mm * n
DO_CALCULATIONS = False

if DO_CALCULATIONS:
    start = perf_counter()
    result = get_projection_matricies(np.linspace(-0.49, 0.49, nn), np.arange(np.pi/(2 * m), np.pi + np.pi / (2 * m), np.pi / m), N, mm)
    print(f"Time taken: {perf_counter() - start}")
    print(result)
    print(result.shape)
    np.save("Code/Projection_matrix.npy", result)
else:
    result = np.load("Code/Projection_matrix.npy")
    print(result)
    print(result.shape)
