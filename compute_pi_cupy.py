import cupy as cp

def compute_pi_cupy(Niter):

    x = cp.random.random(Niter)
    y = cp.random.random(Niter)

    inside = (x*x + y*y) < 1.0

    count = cp.sum(inside)

    piEst = 4.0 * count / Niter

    return float(piEst.get())

