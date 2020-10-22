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
Imports data from a batch csv with format (t, q1, q2, q3, p1, p2, p3, K, V)

Output:
    n batches with following matrices:
        t: time (I, 1)
        Y_p: kinetic variables p1, p2, p3 (3, I)
        Y_q: potential variables q1, q2, q3 (3, I)
        c_p: total kinetic energy K (I, 1)
        c_q: total potential energy V (I, 1)

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
        batch["c_p"] = batch_data[:, 7, np.newaxis] 
        batch["c_q"] = batch_data[:, 8, np.newaxis] # potential energy
        
        batches[i] = batch

    return batches
    

def optimization_step(c, Y, th, d_0, d_k, K, h):
    
    tau = 1.0
    J, dJ = grad_J(c, Y, th, d_0, d_k, K, h)
            
    for key in th:
        th[key] -=  tau*dJ[key]
       
    return th, J

def train(batches, th_q, th_p, d_0, d_k, K, h):
    
    
    tol = 1e-5
    err = 1
    
    maxitr = 10000
    itr = 0
    
    
    for i in range(100):
    #while itr <= maxitr and err > tol:
        for index in batches:
            Y_p = batches[index]["Y_p"]
            c_p = batches[index]["c_p"]
            
            Y_q = batches[index]["Y_q"]            
            c_q = batches[index]["c_q"]
            
            th_q, J_q = optimization_step(c_q, Y_q, th_q, d_0, d_k, K, h)
            th_p, J_p = optimization_step(c_p, Y_p, th_p, d_0, d_k, K, h)
            
        if (i%10 == 0):
            print("i:", i , "J_q: ",  J_q ,"J_p: ", J_p)
            
    return th_q, th_p
        

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
    d_0 = 3
    d_k = d_0    

    batches = import_batches()
    th_p = initialize_weights(d_0, d_k, K)
    th_q = initialize_weights(d_0, d_k, K)
   
    train( batches, th_q, th_p, d_0, d_k, K, h)
    






main()

    