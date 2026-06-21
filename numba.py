from numba import njit, prange, cuda


@njit
def compute_pi_numba_njit(Niter):

    count = 0

    for i in range(Niter):

        x = np.random.random()
        y = np.random.random()

        if x*x + y*y < 1.0:
            count += 1

    piEst = 4.0 * count / Niter

    return piEst


