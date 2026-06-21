@njit(parallel=True)
def compute_pi_numba_parallel(Niter):

    count = 0

    for i in prange(Niter):

        x = np.random.random()
        y = np.random.random()

        if x*x + y*y < 1.0:
            count += 1

    piEst = 4.0 * count / Niter

    return piEst
    
res = compute_pi_numba_parallel(100) # the function gets compiled.

