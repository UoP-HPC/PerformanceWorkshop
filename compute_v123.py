
def compute_pi_baseline_v1(Niter):
   
    
    count=0
    for i in range(Niter):
        x = random.random()
        y = random.random()
        if x**2+y**2 < 1:
            count+=1

    piEst=4*count/Niter
    
    return piEst

def compute_pi_baseline_v2(Niter):
   
    rng = random.random
    
    count=0
    for i in range(Niter):
        x = rng()
        y = rng()
        if x**2+y**2 < 1:
            count+=1

    piEst=4*count/Niter
    
    return piEst
    
def compute_pi_baseline_v3(Niter):
   
    rng = random.random
    
    count=0
    for i in range(Niter):
        x = rng()
        y = rng()
        if x*x+y*y < 1: # x*x is faster than x**2 
            count+=1

    piEst=4*count/Niter
    
    return piEsti
