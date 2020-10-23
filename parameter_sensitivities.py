#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:32:12 2020

@author: isakh
"""
from nn_2 import *
    

def tau():
        
    K = 10
    h = 1/10
    d_0 = 2
    d = 4
    I = 20
    max_it = 200
    
                
    b = generate_synthetic_batches(I)
    
    #c = b["c"]
    #Y = b["Y"]
    c = scale(b["c"])
    Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = [3, 5, 2.5, 2, 1, 0.5, 0.1, 0.01]
    #var = np.arange(2, 40, 6)
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        
        tau = var[i]
        th = initialize_weights(d_0, d, K)
        JJ = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it)
        plt.plot(it, JJ, label="layer: "+ str(tau))
    
    #plt.yscale("log")
    plt.legend()
    plt.show()
    

def layers():
        
    K = 14
    h = 1/10
    d_0 = 2
    d = 4
    I = 20
    max_it = 200
    tau = 0.1
                
    b = generate_synthetic_batches(I)
    
    #c = b["c"]
    #Y = b["Y"]
    c = scale(b["c"])
    Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = np.arange(2, 40, 6)
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        K = var[i]
        th = initialize_weights(d_0, d, K)
        JJ = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it)
        plt.plot(it, JJ, label="layer: "+ str(var[i]))
    
    #plt.yscale("log")
    plt.legend()
    plt.show()

layers()