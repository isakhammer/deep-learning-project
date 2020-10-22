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
    
def gradient_descent(batches, th, d_0, d_k, K, h, name, epochs=1):
    Y_key, c_key = None, None
    
    if name=="q":
        Y_key = "Y_q"
        c_key = "c_q"
    
    if name=="p":
        Y_key = "Y_p"
        c_key = "c_p"
        
    tau = 1.0
    
    for i in range(epochs):
    #while itr <= maxitr and err > tol:
    
        for index in batches:
            Y = batches[index][Y_key]
            c = batches[index][c_key]
            
            J, dJ = grad_J(c, Y, th, d_0, d_k, K, h)
            
            for key in th:
                th[key] -= tau*dJ[key]
                
           
    return th, J

def adams_method(batches, th, d_0, d_k, K, h, name, epochs=1):
    
    Y_key, c_key = None, None
    
    if name=="q":
        Y_key = "Y_q"
        c_key = "c_q"
    
    if name=="p":
        Y_key = "Y_p"
        c_key = "c_p"
        
    
    v = {} 
    m = {}
    
    beta_1, beta_2 =  0.9, 0.999
    alpha, epsilon = 0.01, 10**(-8)
    
    for key in th:
        v[key] = np.zeros(th[key].shape)   
        m[key] = np.zeros(th[key].shape)   
    
    
    for i in range(epochs):
    #while itr <= maxitr and err > tol:
    
        for index in batches:
            Y = batches[index][Y_key]
            c = batches[index][c_key]
            
            j = (i* len(batches) + index)
            
            J, dJ = grad_J(c, Y, th, d_0, d_k, K, h)
            
            for key in th:
                g = dJ[key]
                m[key] = beta_1*m[key] + (1- beta_1)*g
                v[key] = beta_2*v[key] + (1 - beta_2)*(g*g)
                mhat = m[key]/(1 - beta_1**j)
                vhat = v[key]/(1 - beta_2**j)
                th[key] -= alpha*mhat/(np.square(vhat) + epsilon)
        
    return th
    

def train(batches, th_q, th_p, d_0, d_k, K, h):
    
    method = "adams"
    
    if method=="adams":
        th_q = adams_method(batches, th_q, d_0, d_k, K, h, name="q", epochs=1)
    
    if method=="gradient_descent":
        th_q = gradient_descent(batches, th_q, d_0, d_k, K, h, name="q", epochs=1)
                
            
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

    