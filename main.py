#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:27:15 2020

@author: isakh
"""
import numpy as np
import os

def import_batches():
    n_batches = 49
    
    data_prefix = "datalist_batch_"
    data_path = os.path.join(os.path.dirname(__file__), "project_2_trajectories")
    
    # get current path
    
    batches = {}
    
    for i in range(n_batches):
        # assemble track import path
        batch_path = os.path.join(data_path, data_prefix + str(i) + ".csv")
        batch = np.loadtxt(batch_path, delimiter=',', skiprows=1)
        batches[i] = batch

    return batches
    

K = 5
h = 1/2

def initialize_weights(d_0, d_k, K):
    th = {}
    
    if (K>1):
        th["W"+str(0)] = 2*np.random.random((d_k, d_0 )) - 1
        th["b"+str(0)] = np.random.random((d_k, 1))
        
        for i in range(1, K):
            th["W"+str(i)] = 2*np.random.random(( d_k, d_k)) - 1
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
    dsigma = {}
    
    Z[0] = Y
    
    I_d = np.identity(d_k)[:,:d_0]
    
    Z_hat= th["W"+str(0)]@Z[0]+th["b"+str(0)]
    Z[1] = I_d@Z[0] + h*sigma(Z_hat, False)
    
    dsigma[0] = sigma( Z_hat, True)
    
    for k in range(2,K+1):
        Z_hat = th["W"+str(k-1)]@Z[k-1]+th["b"+str(k-1)]
        Z[k] = Z[k-1] + h*sigma(Z_hat, False)
        dsigma[k-1] = sigma(Z_hat, True)
    
    #Z_hat = th["W"+str(K)]@Z[K]+th["b"+str(K)]
    #dsigma[K] = sigma(Z_hat, True)
    
    Upsilon = eta(Z[K].T@th["w"]+th["mu"])
    dUpsilon = eta(Z[K].T@th["w"]+th["mu"], derivative=True)
    
    # Maybe return dZ instead of deta and eta?
    return Z, Upsilon, dUpsilon, dsigma

def grad_J(c ,Y, th, d_0, d_k):
    
    
    I =  Y.shape[1]
    
    # Equation 5
    Z, Upsilon, dUpsilon, dsigma = F_tilde(Y, th, d_0, d_k)    
    
    # Equation 6
    J = 0.5*np.linalg.norm(Upsilon - c)**2

    # Initializing gradient
    dJ = {}

    # Equation (8)  
    dJ["mu"]   =  dUpsilon.T@(Upsilon - c)
        
    # Equation (9)
    dJ["w"] = Z[K]@((Upsilon - c)* dUpsilon)
        
    # Equation (10)
    P = np.zeros(( K+1, d_k, I))
    P[-1] = th["w"] @ ((Upsilon - c)* dUpsilon).T
        
    # Equation 11
    for k in range(K-1,-1,-1):
        P[k] = P[k+1] + h*th["W"+str(k)].T @ (dsigma[k] * P[k+1])  
    
    for k in range(0, K):
        # Equation 12
        dJ["W"+str(k)] = h*(P[k+1] * dsigma[k])@(Z[k]).T
        
        # Equation 13
        dJ["b"+str(k)] = h*(P[k+1] * dsigma[k])@np.ones((I,1))
    
    return J, dJ

def train(c, Y, th, d_0, d_k):
    
    tau = 0.5
    
    tol = 1e-5
    err = 1
    
    maxitr = 10000
    itr = 0
    
    I =  Y.shape[1]
    
    
    for i in range(10000):
    #while itr <= maxitr and err > tol:
            
        J, dJ = grad_J(c, Y, th, d_0, d_k)
        
        for key in th:
            th[key] -=  tau*dJ[key]
        if (i%100):
            print("i:", i , "J ",  J)
            
    return th
        

    
def main():
    # (dxI)
    Y = np.array([
        [1,1,1],        
        [1,1,2],        
        [1,1,1],        
        [1,2,1]        
        ]).T
    
    # (Nx1)
    c = np.array([[1,1,1,1]]).T
    
    d_0 = Y.shape[0]
    d_k = d_0    

    th = initialize_weights(d_0, d_k, K)
   
    train(c, Y, th, d_0, d_k)
    Z, Upsilon, dUpsilon, dsigma =  F_tilde(Y, th, d_0, d_k) 
    print("True ", c.T ," Estimated " ,  Upsilon.T )

import_data()
    
#main()
    
    
