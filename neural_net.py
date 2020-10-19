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
        Th["Th"+str(K)]["W"] = 2*np.random.random((d_k, 1 )) - 1
        Th["Th"+str(K)]["b"] = np.random.random((1, 1))
    else:
        Th["Th"+str(K)] = {}
        Th["Th"+str(K)]["W"] = 2*np.random.random((d_0, 1 )) - 1
        Th["Th"+str(K)]["b"] = np.random.random((1, 1))
    
    return Th


def sigma(x, derivative=False):
  if (derivative):
    return 1 / np.cosh(x)**2 
  return np.tanh(x)

def mu(x, derivative=False):
  if (derivative):
    return (1/4)*(1 / np.cosh(z)**2 )
  return 0.5*(1 + np.tanh(x*0.5))


def Phi_k(Z_k, Th_k):
    return Z_k +  sigma(Z_k@Th_k["W"] + Th_k["b"] )
    


def forward_propogation(Y, Th, h):
    Phi_k(Y, Th["Th"+str(1)])


Y = np.array([
[1,1,1],        
[1,1,1],        
[1,1,1]        
])


d_0 = 3
d_k = 2    
K = 1
h= 1


Th = initialize_weights(d_0, d_k, K)


forward_propogation(Y, Th, h)

print(Th)