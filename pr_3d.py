import os
import random
import scipy.io
import numpy as np
import PhaseRetrieval
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.ndimage import rotate
# %%

seed = int(os.environ["SLURM_ARRAY_TASK_ID"]) #int(sys.argv[-1])
random.seed(seed) 

#%%

def padding(vol):
    raw_size = vol.shape[-1]
    size = 2*raw_size
    padded = np.zeros((size,size,size),dtype = np.float64)
        
    padded[int(0.5*raw_size):int(1.5*raw_size),int(0.5*raw_size):int(1.5*raw_size),int(0.5*raw_size):int(1.5*raw_size)] = vol
    
    return padded


def crop(vol): 
    size = vol.shape[-1]//2
    cropped = np.zeros((size,size,size),dtype = np.float64)
        
    cropped = vol[int(0.5*size):int(1.5*size),int(0.5*size):int(1.5*size),int(0.5*size):int(1.5*size)] 
    
    return cropped

#%%

def fun(vol,target):
        
    
    #################
    # Poission Noise addition #
    #################
    
    # norm = np.max(vol)
    # nphoton = 10
    # vol = vol*nphoton/norm    
    # vol = np.random.poisson(vol)*norm/nphoton
    
    
    # ctr = vol.shape[-1]//2
    # vol /= vol[ctr,ctr,ctr]
    #################
    
    chioduck = PhaseRetrieval.PhaseRetrieval(vol)
    chioduck.CHIO(n_iter=niter) 
    
    image, error, best = chioduck.calc_real_space_error(target,plot=False,return_meta=True) 

    return image,error, best


#%%

niter = 1000         #Number of iteration of phase retrieval process

den = np.load('3d_den.npy')  

# print(f'Max value of density: {np.max(den)}',flush=True)
# den /= np.max(den)
den = padding(den)   

#%%

data = np.abs(np.fft.fftshift(np.fft.fftn(den)))

### rotation ####
# data = rotate(data,angle=45, reshape=False)

#### Hole at center #####
# ctr = data.shape[-1]//2
# data[ctr-5:ctr+6,ctr-5:ctr+6,ctr-5:ctr+6] = data[ctr+6,ctr+6,ctr+6]

# print(f'Max value of diffraction: {np.max(data)}',flush=True)
# data /= np.max(data)

############# Cut-off ########## 
# data = data*(data >= 1e-2*np.max(data))

# ctr = data.shape[-1]//2
# print(f"center val : {data[ctr,ctr,ctr]}")
# print(f"max val : {np.max(data)}")      
      
# if data[ctr,ctr,ctr] > 1: data /= data[ctr,ctr,ctr]


#%%

############## cropping of diff image ##################

#### 15966 #####
# data = data[28:284,28:284,28:284] 
# data = padding(data)

#### ribosome #####
# data = crop(data)
# data = padding(data)

#%%
print('Starting Phase Retrieval...',flush=True)
start = timer()  # record time at the start

pr_recon, pr_error, pr_best = fun(data,den)  

stop = timer()    
print("Time taken: %.3f secs" % (stop - start))

#%%
print(f'min error: {np.min(pr_error)}')
np.save(f'pr_recon_{seed}',pr_recon)
np.save(f'pr_error_{seed}',pr_error)
scipy.io.savemat(f'pr_recon_{seed}.mat',{f"pr_recon_{seed}":pr_recon})

#%%

