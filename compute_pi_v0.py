import random
import time
import math

def MCiter():
    x = random.random()
    y = random.random()
    count=0
    if x**2+y**2 < 1:
            count=1
    return count 

def compute_pi_baseline_v0(Niter):
   
    
    count=0
    for i in range(Niter):
        count+=MCiter() # function call inside loop

    piEst=4*count/Niter
    
    return piEst




Niter = 5_000_000

start = time.perf_counter()
piEst=compute_pi_baseline_v0(Niter)
end = time.perf_counter()
timeTaken = end - start

print(f"Time taken for v0 {Niter} : {timeTaken} s")
print(f"Estimate for pi is {piEst} after {timeTaken} seconds")
print(f"Absolute error is {abs(piEst-math.pi)}")
print(f"speed = {Niter/timeTaken/1e6} million samples per second\n",)



%timeit  compute_pi_baseline_v0(Niter)




C = abs(piEst-math.pi)*math.sqrt(Niter)

timeEstimate_target_prec = timeTaken * (C/target_prec)**2/Niter / 3600/24 # days

