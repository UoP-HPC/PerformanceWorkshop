# kernel must be killed to run this.
import time
import sys
import os

verbose=True

# without that line the default chosen by python would be the number of cores actually available in your jupyter session !
os.environ["OMP_NUM_THREADS"] = "64"

nthreads=os.environ['OMP_NUM_THREADS']
#nthreads = 'unknown'
import numpy as np
N = 2000
# Create two large random matrices
a = np.random.randn(N, N)
b = np.random.randn(N)
Mem = (a.nbytes + b.nbytes)

# matrix vector mult
nrep=100
t1 = time.time()
for i in range(nrep):
    b= a@b

dt = (time.time()-t1)/1000

if verbose == True:
    print(f"Number of threads used: {nthreads}")
    print(f"Memory used {(a.nbytes + b.nbytes)*1e-6}MB")
    
    print(f"Time to 1 matrix vector multiplication {dt/nrep} seconds")

#print("N,nthreads,Mem, matrix vector time")
print(f"{N},{nthreads},{Mem},{dt/nrep}")

