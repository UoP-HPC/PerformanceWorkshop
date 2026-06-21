#without restarting kernel : 
import time
import numpy as np

from threadpoolctl import threadpool_limits
from threadpoolctl import threadpool_info

# ------------------------------------------------
# Problem size
# ------------------------------------------------

N = 2000

a = np.random.randn(N, N)
b = np.random.randn(N)

# ------------------------------------------------
# Function to benchmark
# ------------------------------------------------

def benchmark(nthreads):

    with threadpool_limits(limits=nthreads):

        nrep = 100
        x = b.copy()
        t1 = time.time()
        for _ in range(nrep):
            x = a @ x

        t2 = (time.time() - t1) / nrep
    
    print(f"Threads: {nthreads}")
    print(f"matvec  : {t2:.6f} s")

# ------------------------------------------------
# Inspect backend
# ------------------------------------------------
print(threadpool_info())

# ------------------------------------------------
# Test different thread counts
# ------------------------------------------------
for nthreads in [1, 2, 4, 8, 16, 32, 64]:
    benchmark(nthreads)
