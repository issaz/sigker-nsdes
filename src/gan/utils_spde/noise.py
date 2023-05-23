import numpy as np
import pandas as pd
import math
import torch
from tqdm import tqdm
from time import time

class Noise(object):
    
    def __init__(self, correlation = 'white', noise_size = 1):
        
        self.noise_size = noise_size
        
        if isinstance(correlation, str):
            if correlation == 'white':
                self.correlation = WN_corr
            elif correlation == 'colored':
                self.correlation = smooth_corr
            else:
                NameError('noise type not recognized')
        else:
            self.correlation = correlation
        
    def return_corr(self, grid):
        T, X = grid[0,:,1], grid[:,0,0]
        
        dt, dx = T[1]-T[0], X[1]-X[0]
        N = len(X)
        
        # Create correlation Matrix in space
        grid_x = X.unsqueeze(-1).repeat(1,len(X))
        space_corr = self.correlation(grid_x, dx * (N))  #(x_k,j)
        
        return space_corr
        
    
    def partition(self, a, b, dx): #makes a partition of [a,b] of equal sizes dx
        return torch.linspace(a, b, int((b - a) / dx) + 1)
    
    # Create l dimensional Brownian motion with time step = dt
    
    def BM(self, n_sample_paths, start, stop, dt, l):
        T = self.partition(start, stop, dt)
        # assign to each point of len(T) time point an N(0, \sqrt(dt)) standard l dimensional random variable
        BM = torch.sqrt(dt)*torch.randn(n_sample_paths, len(T), l, device=dt.device)
        BM[:,0] = 0 #set the initial value to 0
        BM = torch.cumsum(BM, axis = 1) # cumulative sum: B_n = \sum_1^n N(0, \sqrt(dt))
        return BM

    def sample(self, n_sample_paths, grid):

        return torch.stack([self._sample(n_sample_paths, grid) for i in range(self.noise_size)], dim=1)
        
    
    # Create space time noise. White in time and with some correlation in space.
    # See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow
    # X here is points in space as in SPDE1 function.
    def _sample(self, n_sample_paths, grid):

        T, X = grid[0,:,1], grid[:,0,0]

        dt, dx = T[1]-T[0], X[1]-X[0]

#         T, X = self.partition(s, t, dt), self.partition(a, b, dx) #time points, space points,
        N = len(X)

        # Create correlation Matrix in space
        grid_x = X.unsqueeze(-1).repeat(1,len(X))

        space_corr = self.correlation(grid_x, dx * N)  #(x_k,j)

        B = self.BM(n_sample_paths, T[0], T[-1], dt, N)
        
        return torch.matmul(B, torch.transpose(space_corr,0,1)).permute(0,2,1)
        
    # save list of noises as a multilevel dataframe csv file
    def save(self, W, name):
        W.to_csv(name)
        
#Correlation function that approximates WN in space.
# See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow

def WN_corr(x, a):
    j = 1.*torch.arange(x.shape[0]).to(x.device)
    return torch.sqrt(2 / a) * torch.sin(j * torch.pi * x / a)

def smooth_corr(x, a, r = 2):
    my_eps=0.001
    j = 1.*torch.arange(1,x.shape[0]+1).to(x.device)
    j[-1] = 0.
    q = j**(-(2*r+1+my_eps)/2)
    res = torch.sqrt(q)*torch.sqrt(2 / a) * torch.sin(j * torch.pi * x / a)
    return res
