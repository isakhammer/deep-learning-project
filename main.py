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
        th["W"+str(0)] = 2*np.random.random((d_0, d_k )) - 1
        th["b"+str(0)] = np.random.random((1, d_k))
        
        for i in range(1,K):
            th["W"+str(i)] = 2*np.random.random((d_k, d_k )) - 1
            th["b"+str(i)] = np.random.random((1, d_k))
            
        th["w"] = 2*np.random.random((d_k, 1 )) - 1
        th["mu"] = np.random.random((1, 1))
    else:
        th["w"] = 2*np.random.random((d_0, 1 )) - 1
        th["mu"] = np.random.random((1, 1))
    
    return th



def main():
    # (dxN)
    Y = np.array([
        [1,1,1],        
        [1,1,2],        
        [1,1,1],        
        [1,2,1]        
        ]).T
    
    # (Nx1)
    c = np.array([[1,1,1,1]]).T
    
    
    d_0 = 3
    d_k = 2    


    th = initialize_weights(d_0, d_k, K)
    print(th)
   
    #train(c, Y, th)
    
main()
    
    
