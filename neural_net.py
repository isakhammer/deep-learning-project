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
            
        Th["w"] = 2*np.random.random((d_k, 1 )) - 1
        Th["mu"] = np.random.random((1, 1))
    else:
        Th["w"] = 2*np.random.random((d_0, 1 )) - 1
        Th["mu"] = np.random.random((1, 1))
    
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
Input:
    Z_k = (d, N)
    w = (d, 1)
    mu = (1, 1)

out: 
    \eta( Z_k^T w  + \mu ) = (I, 1) 
"""
def F_tilde(Z_k, Th):
    Z = {}
    
    Z_K =  Z_k
    Z["K"] = Z_K
    
    eta_    = eta( Z_K.T@Th["w"]  + Th["mu"], derivative = False )
    deta_ = eta( Z_K.T@Th["w"]  + Th["mu"], derivative = True )
    
    # Maybe return dZ instead of deta and eta?
    return eta_, deta_,  Z


"""
Input 
    Y: (d, I)
    Th: weights.
    h: stepsize
    K: number of layers

Output:
    Upsilon: (I, 1)    
"""
def forward_propogation(Y, Th, h, K):
    upsilon, Z = F_tilde(Y, Th)    
    return upsilon, Z


def train(c, Y, Th, h, K):
    
    
    tau = 0.5
    for i in range(10000):     
        
        
        F_t, dF_t, Z = F_tilde(Y, Th )

        # Equation (8)  
        upsilon = F_t
        dJ_mu   =  F_t.T@(upsilon - c)
        
        # Equation (9)
        dJ_w = (upsilon - c)* dF_t
        dJ_w = Z["K"]@dJ_w
        
        
        Th["w"] -= tau*dJ_w
        Th["mu"] -= tau*dJ_mu
        print(np.linalg.norm(upsilon - c) )
    
    #dJ_w   =  Y@(upsilon - c) @( F(Y, Th["Th"+str(K)], derivative=True))
    
    #print(J_mu)
    #print( "f",  f.shape,"c" ,upsilon.shape)
    #Y@(upsilon - c) 
    return upsilon

    

def single_neuron():
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
    K   = 1
    h   = 1


    Th = initialize_weights(d_0, d_k, K)
   
    train(c, Y, Th, h, K)
    
    



    
single_neuron()