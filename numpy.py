
import random
import numpy as np

from numba import njit, prange, cuda


# ============================================================
# 1. BASELINE PYTHON
# ============================================================

def compute_pi_baseline_v3(Niter):

    rng = random.random

    count = 0

    for i in range(Niter):

        x = rng()
        y = rng()

        if x*x + y*y < 1.0:
            count += 1

    piEst = 4.0 * count / Niter

    return piEst


# ============================================================
# 2. NUMBA NJIT (nopython=True)
# ============================================================

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


# ============================================================
# 3. NUMBA OBJECT MODE (nopython=False)
# ============================================================

@njit(nopython=False)
def compute_pi_numba_object_mode(Niter):

    rng = random.random

    count = 0

    for i in range(Niter):

        x = rng()
        y = rng()

        if x*x + y*y < 1.0:
            count += 1

    piEst = 4.0 * count / Niter

    return piEst


# ============================================================
# 4. NUMBA PARALLEL CPU
# ============================================================

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


# ============================================================
# 5. NUMBA CUDA GPU
# ============================================================

@cuda.jit
def compute_pi_cuda_kernel(rng_states, results):

    idx = cuda.grid(1)

    if idx < results.size:

        # xoroshiro random numbers
        x = cuda.random.xoroshiro128p_uniform_float32(rng_states, idx)
        y = cuda.random.xoroshiro128p_uniform_float32(rng_states, idx)

        if x*x + y*y < 1.0:
            results[idx] = 1
        else:
            results[idx] = 0


def compute_pi_numba_cuda(Niter):

    from numba.cuda.random import (
        create_xoroshiro128p_states
    )

    threads_per_block = 256
    blocks = (Niter + threads_per_block - 1) // threads_per_block

    rng_states = create_xoroshiro128p_states(
        threads_per_block * blocks,
        seed=42
    )

    results = np.zeros(Niter, dtype=np.int32)

    d_results = cuda.to_device(results)

    compute_pi_cuda_kernel[blocks, threads_per_block](
        rng_states,
        d_results
    )

    results = d_results.copy_to_host()

    count = results.sum()

    piEst = 4.0 * count / Niter

    return piEst

# ============================================================
# 6. PURE NUMPY VECTORIZED VERSION
# ============================================================

def compute_pi_numpy(Niter):

    x = np.random.random(Niter)
    y = np.random.random(Niter)

    inside = (x*x + y*y) < 1.0

    count = np.sum(inside)

    piEst = 4.0 * count / Niter

    return piEst


# ============================================================
# 7. NUMPY + NUMBA
# ============================================================

@njit
def compute_pi_numpy_numba_kernel(x, y):

    count = 0

    for i in range(x.size):

        if x[i]*x[i] + y[i]*y[i] < 1.0:
            count += 1

    return count


def compute_pi_numpy_numba(Niter):

    # NumPy generates vectors
    x = np.random.random(Niter)
    y = np.random.random(Niter)

    # Numba accelerates the loop
    count = compute_pi_numpy_numba_kernel(x, y)

    piEst = 4.0 * count / Niter

    return piEst


# ============================================================
# 8. NUMPY + NUMBA PARALLEL
# ============================================================

@njit(parallel=True)
def compute_pi_numpy_numba_parallel_kernel(x, y):

    count = 0

    for i in prange(x.size):

        if x[i]*x[i] + y[i]*y[i] < 1.0:
            count += 1

    return count


def compute_pi_numpy_numba_parallel(Niter):

    x = np.random.random(Niter)
    y = np.random.random(Niter)

    count = compute_pi_numpy_numba_parallel_kernel(x, y)

    piEst = 4.0 * count / Niter

    return piEst



