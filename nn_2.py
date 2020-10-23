#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:29:57 2020

@author: isakh
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_batches(I):
    
    batch = {} 
    
    d_0 = 2
    
    
    batch["Y"] = np.array([
            [1,1],
            [2,3],
            [1,-2]])
        
   # batch["Y"] = np.random.uniform(high=2, low=-2, size=(d_0,I) )    
    
    batch["c"] = 0.5*batch["Y"][0,:]**2 + 0.5*batch["Y"][1,:]**2
    batch["c"] = batch["c"][:, np.newaxis]
        
    return batch


def F_tilde(Y, th, d_0, d, K, h):
    
    Z = {}
    I_d = np.identity(d)[:,:d_0]
    Z[0] = I_d@Y

    for k in range(K):
        Z_hat = th["W"+str(k)]@Z[k]+th["b"+str(k)]
        Z[k+1] = Z[k] + h*sigma(Z_hat, False)
    
    Upsilon = eta(Z[K].T@th["w"]+th["mu"])
    
    return Z, Upsilon 


def initialize_weights(d_0, d, K):
    th = {}
    
    for i in range(K):
        th["W"+str(i)] = np.identity(d)
        th["b"+str(i)] = np.zeros((d, 1))
            
    th["w"] = np.ones((d, 1 ))
    th["mu"] = np.zeros((1, 1))
    
    return th



def sigma(x, derivative=False):   
    if (derivative):
        return 1 / np.cosh(x)**2 
    return np.tanh(x)

def eta(x, derivative=False, identity=True):
    if identity==True:
        if (derivative):
            return np.ones(x.shape)
        return x
    else:
        if (derivative):
            return 0.25*(np.cosh(0.5*x) )**(-2)
        return 0.5*np.tanh(0.5*x) + 0.5
        


def J_func(Upsilon, c):
    return 0.5*np.linalg.norm(c - Upsilon)**2


def scale(x, alpha=0, beta=1):
    
    a = np.min(x)
    b = np.max(x)
    return ( (b - x)*alpha + (x - a)*beta)/(b - a)



def gradientDesent(K, th, dJ_w, dJ_mu, dJ_W, dJ_b, tau):
    
    th["mu"] = th["mu"] - tau*dJ_mu
    th["w"] = th["w"] - tau*dJ_w
    for k in range(K):
        th["W"+str(k)] = th["W"+str(k)] -  tau*dJ_W[k]
        th["b"+str(k)] = th["b"+str(k)] -  tau*dJ_b[k]
    
    return th

def adams(K, th, dJ_w, dJ_mu, dJ_W, dJ_b, tau):
    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.01
    epsilon = 1e-8
    v0 = 0
    m0 = 0
    
    
    return th



def train(c, d, d_0, K, h, Y, th, tau=0.0005, max_it=60):
    # compute Zk
    err = np.inf
    tol = 10**(-3)
    
    
    
    itr = 0
    
    JJ = np.zeros(max_it)
    
    while (itr < max_it ):
        
        Z, Upsilon = F_tilde(Y, th, d_0, d, K, h)
        I = Upsilon.shape[0]
        
        
        etahat = eta(Z[K].T@th["w"] + th["mu"]*np.ones(( I, 1)), derivative=True )
        
        # Equation (10)
        P = np.zeros(( K+1, d, I))
        P[K] = np.outer(th["w"], ( (Upsilon - c)* etahat).T)
        
        dJ_mu = etahat.T @(Upsilon - c)
        
        dJ_w = Z[K] @ ((Upsilon - c) * etahat)
        
        for k in range(K, 0, -1):
            P[k-1] = P[k] + h*th["W"+str(k-1)].T @ (sigma(th["W"+str(k-1)]@Z[k-1]+np.outer(th["b"+str(k-1)],np.ones(I)), True) * P[k])
            
        dJ_W = np.zeros((K, d, d))
        dJ_b = np.zeros((K, d, 1))
        
        for k in range(K):
            dsigma = sigma(th["W"+str(k)]@Z[k]+np.outer(th["b"+str(k)],np.ones(I)),True)
            
            dJ_W[k] = h*(P[k+1]*dsigma) @ Z[k].T
            dJ_b[k] = (h*(P[k+1]*dsigma) @ np.ones(I))[:,np.newaxis]
            
        th = gradientDesent(K, th, dJ_w, dJ_mu, dJ_W, dJ_b, tau)
        
        err = J_func(Upsilon,c)  
        
        JJ[itr] = err
        
        itr += 1
        
        
        
    return JJ
        
        
    

def main():
        
    K = 10
    h = 1/100
    d_0 = 2
    d = 4
    I = 20
    max_it = 10000
    
                
    b = generate_synthetic_batches(I)
    c = b["c"]
    c = scale(b["c"])
    Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    
    KK = np.arange(1,4, 1)
    it = np.arange(max_it)
    
    for i in range(len(KK)):    
        th = initialize_weights(d_0, d, KK[i])
        JJ = train(c, d, d_0, KK[i], h, Y, th, tau=0.0005, max_it=max_it)
        plt.plot(it, JJ, label="layer: "+ str(KK[i]))
    
    plt.yscale("log")
    plt.legend()
    plt.show()
    
    
main()