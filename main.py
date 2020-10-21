#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:27:15 2020

@author: isakh
"""

import numpy as np
K = 5
h = 1/2

def initialize_weights(d_0, d_k, K):
    th = {}
    
    if (K>1):
        th["W"+str(0)] = 2*np.random.random((d_k, d_0 )) - 1
        th["b"+str(0)] = np.random.random((d_k, 1))
        
        for i in range(1,K+1):
            th["W"+str(i)] = 2*np.random.random((d_k, d_k )) - 1
            th["b"+str(i)] = np.random.random((d_k, 1))
            
        th["w"] = 2*np.random.random((d_k, 1 )) - 1
        th["mu"] = np.random.random((1, 1))
    else:
        th["w"] = 2*np.random.random((d_0, 1 )) - 1
        th["mu"] = np.random.random((1, 1))
    
    return th

def sigma(x, derivative=False):
  if (derivative):
    return 1 / np.cosh(x)**2 
  return np.tanh(x)

def eta(x, derivative=False):
  if (derivative):
    return (1/4)*(1 / np.cosh(x)**2 )
  return 0.5*(1 + np.tanh(x*0.5))


"""
Input:
    Z_k = (d, N)
    w = (d, 1)
    mu = (1, 1)

out: 
    \eta( Z_k^T w  + \mu ) = (I, 1) 
"""
def F_tilde(Y, th, d_0, d_k):
    
    Z = {}
    Z[0] = Y
    I_d = np.identity(d_k)[:,:d_0]
    print(I_d.shape)
    Z[1] = I_d@Z[0] + h*sigma(th["W"+str(0)]@Z[0]+th["b"+str(0)], False)
    for k in range(2,K+1):
        Z[k] = Z[k-1] + h*sigma(th["W"+str(k-1)]@Z[k-1]+th["b"+str(k-1)], False)
        print(k,Z[k].shape)
        
    Upsilon = eta(Z[K].T@th["w"]+th["mu"])
    print(Upsilon.shape)

    #eta_    = eta( Z[K].T@th["w"]  + th["mu"], derivative = False )
    #deta_ = eta( Z[K].T@th["w"]  + th["mu"], derivative = True )
    # Maybe return dZ instead of deta and eta?
    return Z,Upsilon

def train(c, Y, th, d_0, d_k):
    
    tau = 0.5
    
    tol = 1e-5
    err = 1
    
    maxitr = 10000
    itr = 0
    
    for i in range(1):
    #while itr <= maxitr and err > tol:
        Z, Upsilon = F_tilde(Y, th, d_0, d_k)
        
        

def main():
    # (dxI)
    Y = np.array([
        [1,1,1],        
        [1,1,2],        
        [1,1,1],        
        [1,2,1]        
        ]).T
    print(Y.shape)
    
    # (Nx1)
    c = np.array([[1,1,1,1]]).T
    
    
    d_0 = 3
    d_k = 6    


    th = initialize_weights(d_0, d_k, K)
   
    train(c, Y, th, d_0, d_k)
    a = F_tilde(Y,th,d_0,d_k)
    
main()
    
    
