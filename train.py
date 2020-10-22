#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:11:30 2020

@author: isakh
"""


import numpy as np
import os
from neural_net import grad_J, F_tilde

"""
Imports data

Output:
    n batches with following matrices:
        t: time (I, 1)
        Y_q: kinetic variables (3, I)
        Y_k: potential variables (3, I)
        c_K: total kinetic energy (I, 1)
        c_V: total potential energy (I, 1)

"""
def import_batches():
    n_batches = 49
    
    data_prefix = "datalist_batch_"
    data_path = os.path.join(os.path.dirname(__file__), "project_2_trajectories")
    
    batches = {}
    
    for i in range(n_batches):
        # assemble track import path
        batch_path = os.path.join(data_path, data_prefix + str(i) + ".csv")
        batch_data = np.loadtxt(batch_path, delimiter=',', skiprows=1)
        
        # np.newaxis is adding a dimension such that (I,) -> (I, 1)
        batch = {}
        batch["t"] = batch_data[:, 0, np.newaxis]
        batch["Y_q"] = batch_data[:, 1:4].T
        batch["Y_p"] = batch_data[:, 4:7].T
        batch["c_K"] = batch_data[:, 7, np.newaxis]
        batch["c_V"] = batch_data[:, 7, np.newaxis]
        
        batches[i] = batch

    return batches
    


def train(c, Y, th, d_0, d_k, K, h):
    
    tau = 0.5
    
    tol = 1e-5
    err = 1
    
    maxitr = 10000
    itr = 0
    
    I =  Y.shape[1]
    
    for i in range(10000):
    #while itr <= maxitr and err > tol:
        print(i)   
        J, dJ = grad_J(c, Y, th, d_0, d_k, K, h)
        
        for key in th:
            th[key] -=  tau*dJ[key]
        if (i%100 == 0):
            print("i:", i , "J ",  J)
            
    return th
        

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


    
def main():
    
    K = 5
    h = 1/2
    
    # (dxI)
    Y = np.array([
        [1,1,1],        
        [1,1,2],        
        [1,1,1],        
        [1,2,1]        
        ]).T
    
    # (Ix1)
    c = np.array([[1,1,1,1]]).T
    print(c.shape, Y.shape)
    
    d_0 = Y.shape[0]
    d_k = d_0    

    th = initialize_weights(d_0, d_k, K)
   
    train(c, Y, th, d_0, d_k, K, h)
    Z, Upsilon, dUpsilon, dsigma =  F_tilde(Y, th, d_0, d_k, K, h)
    print("True ", c.T ," Estimated " ,  Upsilon.T )






batches = import_batches()
main()

    