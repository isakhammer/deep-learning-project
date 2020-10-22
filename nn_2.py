#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:29:57 2020

@author: isakh
"""

import numpy as np


def generate_synthetic_batches():
    
    batch = {}   
    I = 4
    
    d = 2
        
    batch["Y"] = np.random.uniform(high=2, low=-2, size=(2,I) )    
    batch["c"] = 0.5*batch["Y"][0,:]**2 + 0.5*batch["Y"][1,:]**2
    batch["c"] = batch["c"][:, np.newaxis]
        
    return batch


def F_tilde(Y, th, d_0, d, K, h):
    
    Z = {}
    I_d = np.identity(d)[:,:d_0]
    Z[0] = I_d@Y

    for k in range(K):
        Z_hat = th["W"+str(k)]@Z[k]+th["b"+str(k)]
        Z[k+1] = Z[k] + h*sigma(Z_hat, False)
    
    Upsilon = eta(Z[K].T@th["w"]+th["mu"])
    
    return Z, Upsilon 


def initialize_weights(d_0, d, K):
    th = {}
    
    for i in range(K):
        th["W"+str(i)] = 0.5*np.ones(( d, d))
        th["b"+str(i)] = 0.5*np.ones((d, 1))
            
    th["w"] = 0.5*np.ones((d, 1 ))
    th["mu"] = 0.5*np.ones((1, 1))
    
    return th



def sigma(x, derivative=False):   
    if (derivative):
        return 1 / np.cosh(x)**2 
    return np.tanh(x)

def eta(x, derivative=False):
    if (derivative):
        return np.ones(x.shape)
    return x
    


def main():
        
    K = 2
    h = 0.5
    d_0 = 2
    d = 4
    
    b = generate_synthetic_batches()
    th = initialize_weights(d_0, d, K)
    
    Z, Upsilon = F_tilde(b["Y"], th, d_0, d, K, h)
    
    
    
    
main()