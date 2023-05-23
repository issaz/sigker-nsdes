import numpy as np
import pandas as pd
import math
import torch
from tqdm import tqdm
from time import time


def get_oned_bj(dtref,J,a,r=0.5,device='cpu'):
    """
    Alg 10.3 Page 442 in the book "An Introduction to Computational Stochastic PDEs"
    """
    myeps=0.001
    jj = torch.cat([torch.arange(1,J//2,device=device), torch.arange(- J//2 + 1,-1,device=device)]) 
    root_qj = torch.arange(len(jj)+1, device=device)
    root_qj[1:] = torch.abs(jj)**(-(2*r+1+myeps)/2)
    bj = root_qj * torch.sqrt(dtref/torch.sqrt(a))*J
    return bj

def get_oned_dW(bj,kappa,M,device):
    """
    Alg 10.4 Page 442 in the book "An Introduction to Computational Stochastic PDEs"
    """
    J = len(bj)
    if (kappa == 1):
        nn = torch.randn(M,J,2,device=device)  
    else:
        nn = torch.sum(torch.randn(kappa,M,J,2,device=device),0)
#         nn = torch.sum(torch.randn(kappa,M,J[0],J[1],2,device=device),0)
    # update real part
    nn[:,J//2+1:,0] = torch.flip(nn[:,1:J//2,0], [1])/np.sqrt(2)
    nn[:,1:J//2,0] *= 1./np.sqrt(2)
    
    # update imaginary part
    nn[:,0,1] = 0.
    nn[:,J//2,1] = 0.
    nn[:,J//2+1:,1] = -torch.flip(nn[:,1:J//2,1], [1])
    nn[...,1] *= 1./np.sqrt(2)
    
    nn = torch.view_as_complex(nn)  # to check if in correct order
    tmp = torch.fft.ifft(bj*nn,dim=-1)
    dW1 = torch.real(tmp)
    return dW1

class Noise(object):
    
    def __init__(self, correlation = None, noise_size = 1):
        
        self.noise_size = noise_size
        
        if correlation is None:
            self.correlation = get_oned_bj
        else:
            self.correlation = correlation
        
    def return_bj(self, grid, kappa=1):
                
        T, X = grid[0,:,1], grid[:,0,0]
        dt, dx = T[1]-T[0], X[1]-X[0]
        assert len(X)%2 == 0, "the number of spatial points should be even"
        
        bj = self.correlation(dt/kappa,len(X),X[-1]-X[0],device=grid.device)
        return bj

    def sample(self, n_sample_paths, grid, kappa=1):
        
        return torch.stack([self._sample(n_sample_paths, grid, kappa, grid.device) for i in range(self.noise_size)], dim=1)
        
    
    # Create space time noise. White in time and with some correlation in space.
    # See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow
    # X here is points in space as in SPDE1 function.
    def _sample(self, n_sample_paths, grid, kappa=1, device='cpu'):
        
        T, X = grid[0,:,1], grid[:,0,0]
        dt, dx = T[1]-T[0], X[1]-X[0]
        assert len(X)%2 == 0, "the number of spatial points should be even"
        
        bj = self.correlation(dt/kappa,len(X),X[-1]-X[0],device=device)
        dWs = get_oned_dW(bj,kappa,n_sample_paths*len(T),device=device)
        dWs = dWs.reshape(n_sample_paths, len(T), len(X))
        return torch.cumsum(dWs,1).permute(0,2,1)
        
        
    # save list of noises as a multilevel dataframe csv file
    def save(self, W, name):
        W.to_csv(name)
        
#Correlation function that approximates WN in space.
# See Example 10.31 in "AN INTRODUCTION TO COMPUTATIONAL STOCHASTIC PDES" by Lord, Powell, Shardlow
