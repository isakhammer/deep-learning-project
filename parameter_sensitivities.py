#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:32:12 2020

@author: isakh
"""
from nn_2 import *
    

def tau_sensitivity():
                
    b = generate_synthetic_batches(I)
    
    c = scale(b["c"])
    Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = [ 2, 1, 0.5, 0.25, 0.1, 0.01]
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        tau = var[i]
        th = initialize_weights(d_0, d, K)
        JJ = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it)
        plt.plot(it, JJ, label="tau: "+ str(tau))
    
    plt.title("Tau Sensitivity Analysis")
    plt.legend()
    plt.show()
    

def layer_sensitivity():
             
    b = generate_synthetic_batches(I)
    
    c = scale(b["c"])
    Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = var = [ 4, 6, 10, 14, 20]
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        K = var[i]
        th = initialize_weights(d_0, d, K)
        JJ = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it)
        plt.plot(it, JJ, label="K: "+ str(var[i]))
    
    plt.title("Layer Sensitivity Analysis")
    plt.legend()
    plt.show()

    

def h_sensitivity():
             
    b = generate_synthetic_batches(I)
    
    c = scale(b["c"])
    Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = var = [ 0.14, 0.12, 0.1, 0.07, 0.05, 0.01]
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        h = var[i]
        th = initialize_weights(d_0, d, K)
        JJ = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it)
        plt.plot(it, JJ, label="h: "+ str(var[i]))
    
    plt.title("h Sensitivity Analysis")
    plt.legend()
    plt.show()


def d_sensitivity():
             
    b = generate_synthetic_batches(I)
    
    c = scale(b["c"])
    Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = var = [ 2, 3, 4, 5,6,7,8, 10 ]
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        d = var[i]
        th = initialize_weights(d_0, d, K)
        JJ = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it)
        plt.plot(it, JJ, label="h: "+ str(var[i]))
    
    #plt.yscale("log")
    plt.title("k Sensitivity Analysis")
    plt.legend()
    plt.show()

# Default values based on the analysis
K = 20
h = 0.1
d_0 = 2
d = 4
I = 20
max_it = 300
tau = 0.25

layer_sensitivity()
tau_sensitivity()
h_sensitivity()
d_sensitivity()