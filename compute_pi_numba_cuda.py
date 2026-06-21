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

