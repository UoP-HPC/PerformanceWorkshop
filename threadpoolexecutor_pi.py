import numpy as np
from concurrent.futures import ThreadPoolExecutor


def worker(n):
    x = np.random.random(n)
    y = np.random.random(n)
    return np.sum(x*x + y*y < 1.0)


def compute_pi_numpy_threads(N, nthreads=8):

    chunk = N // nthreads

    with ThreadPoolExecutor(max_workers=nthreads) as pool:
        counts = list(pool.map(worker, [chunk]*nthreads))

    count = sum(counts)

    return 4.0 * count / (chunk * nthreads)
