# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:47:13 2020

@author: isakh
"""
import numpy as np

def initialize_weights(d_0, d_k, K):
    Th = {}
    
    if (K>1):
        Th["Th"+str(1)] = {}
        Th["Th"+str(1)]["W"] = 2*np.random.random((d_0, d_k )) - 1
        Th["Th"+str(1)]["b"] = np.random.random((1, d_k))
        
        for i in range(1,K):
            Th["Th"+str(i)] = {}
            Th["Th"+str(i)]["W"] = 2*np.random.random((d_k, d_k )) - 1
            Th["Th"+str(i)]["b"] = np.random.random((1, d_k))
            
        Th["Th"+str(K)] = {}
        Th["Th"+str(K)]["w"] = 2*np.random.random((d_k, 1 )) - 1
        Th["Th"+str(K)]["mu"] = np.random.random((1, 1))
    else:
        Th["Th"+str(K)] = {}
        Th["Th"+str(K)]["w"] = 2*np.random.random((d_0, 1 )) - 1
        Th["Th"+str(K)]["mu"] = np.random.random((1, 1))
    
    return Th


def sigma(x, derivative=False):
  if (derivative):
    return 1 / np.cosh(x)**2 
  return np.tanh(x)

def eta(x, derivative=False):
  if (derivative):
    return (1/4)*(1 / np.cosh(x)**2 )
  return 0.5*(1 + np.tanh(x*0.5))


def Phi_k(Z_k, Th_k):
    return Z_k +  sigma( Z_k@Th_k["W"] + Th_k["b"] )
    
"""
Z_k = (d, N)
w = (d, 1)
mu = (1, 1)

out: (1xN)
"""
def F(Z_k, Th_K, derivative=False):
    return eta( Z_k.T @ Th_K["w"] + Th_K["mu"], derivative )


def forward_propogation(Y, Th, h, K):
    upsilon = F(Y, Th["Th"+str(K)])    
    print("ups", upsilon.shape,
          "Y", Y.shape)
    return upsilon



def train(c, Y, Th, h, K):
    upsilon = forward_propogation(Y, Th, h, K)
    
    print(F(Y, Th["Th"+str(K)], derivative=(True)).shape,
          "\n",
          (upsilon - c).shape, 
          "\n",
          Y.shape)
    
    dJ_mu   =  F(Y, Th["Th"+str(K)], derivative=(True)).T @(upsilon - c)
    
    Y@(upsilon - c) 
    
    #dJ_w   =  Y@(upsilon - c) @( F(Y, Th["Th"+str(K)], derivative=True))
    
    return upsilon

    

def single_neuron():
    Y = np.array([
        [1,1,1],        
        [1,1,2],        
        [1,1,1],        
        [1,2,1]        
        ]).T
    
    c = np.array([
        [1],        
        [1],        
        [1],        
        [1]  
        ])
    

    d_0 = 3
    d_k = 2    
    K = 1
    h= 1


    Th = initialize_weights(d_0, d_k, K)
   
    train(c, Y, Th, h, K)
    
    
    
    
single_neuron()