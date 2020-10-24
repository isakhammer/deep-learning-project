#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:29:57 2020

@author: isakh
"""

import numpy as np

from data import generate_synthetic_batches


def F_tilde(Y, th, d_0, d, K, h):
    
    Z = {}
    I_d = np.identity(d)[:,:d_0]
    Z[0] = I_d@Y

    for k in range(K):
        Z_hat = th["W"][k]@Z[k]+th["b"][k]
        Z[k+1] = Z[k] + h*sigma(Z_hat, False)
    
    Upsilon = eta(Z[K].T@th["w"]+th["mu"])
    
    return Z, Upsilon 


def initialize_weights(d_0, d, K):
    th = {}
    
    th["W"] = np.zeros((K, d, d))
    th["b"] = np.zeros((K, d, 1))
    
    for i in range(K):
        th["W"][i] = np.identity(d)
        th["b"][i] = np.zeros((d, 1))
            
    th["w"] = np.ones((d, 1 ))
    th["mu"] = np.zeros((1, 1))
    
    return th



def sigma(x, derivative=False):   
    if (derivative):
        return 1 / np.cosh(x)**2 
    return np.tanh(x)

def eta(x, derivative=False, identity=False):
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


""""
def gradientDesent(K, th, dJ_w, dJ_mu, dJ_W, dJ_b, tau):
    
    th["mu"] = th["mu"] - tau*dJ_mu
    th["w"] = th["w"] - tau*dJ_w
    for k in range(K):
        th["W"][k] = th["W"][k] -  tau*dJ_W[k]
        th["b"][k] = th["b"][k] -  tau*dJ_b[k]
    
    return th
"""
def gradientDesent(K, th, dJ, tau):
    
    th["mu"] = th["mu"] - tau*dJ["mu"]
    th["w"] = th["w"] - tau*dJ["w"]
    for k in range(K):
        th["W"][k] = th["W"][k] -  tau*dJ["W"][k]
        th["b"][k] = th["b"][k] -  tau*dJ["b"][k]
    
    return th


def adams(K, th, dJ_w, dJ_mu, dJ_W, dJ_b, tau):
    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.01
    epsilon = 1e-8
    v0 = 0
    m0 = 0
    
    return th

def n(x):
    return np.linalg.norm(x)

def dJ_func(c, Y, th, d_0, d, K, h):
    Z, Upsilon = F_tilde(Y, th, d_0, d, K, h)
    I = Upsilon.shape[0]
        
    etahat = eta(Z[K].T@th["w"] + th["mu"]*np.ones(( I, 1)), derivative=True )
        
    P = np.zeros(( K+1, d, I))
    P[K] = np.outer(th["w"], ( (Upsilon - c)* etahat).T)
        
    dJ_mu = etahat.T @(Upsilon - c)
        
    dJ_w = Z[K] @ ((Upsilon - c) * etahat)
        
    for k in range(K, 0, -1):
        P[k-1] = P[k] + h*th["W"][k-1].T @ (sigma(th["W"][k-1]@Z[k-1]+np.outer(th["b"][k-1],np.ones(I)), True) * P[k])
            
    dJ_W = np.zeros((K, d, d))
    dJ_b = np.zeros((K, d, 1))
        
    for k in range(K):
        dsigma = sigma(th["W"][k]@Z[k]+np.outer(th["b"][k],np.ones(I)),True)
            
        dJ_W[k] = h*(P[k+1]*dsigma) @ Z[k].T
        dJ_b[k] = (h*(P[k+1]*dsigma) @ np.ones(I))[:,np.newaxis]
    dJ = {}
    dJ["w"], dJ["mu"], dJ["W"], dJ["b"] = dJ_w, dJ_mu, dJ_W, dJ_b
#    return dJ_w, dJ_mu, dJ_W, dJ_b
    return dJ

def train(c, d, d_0, K, h, Y, th, tau=0.0005, max_it=60, print_it=False, method="gd"):
    # compute Zk
    err = np.inf
    tol = 10**(-3)
    
    
    itr = 0
    Z, Upsilon = F_tilde(Y, th, d_0, d, K, h)
    JJ = np.zeros(max_it+1)
    err = J_func(Upsilon,c)
    
    JJ[0] = err
    
    # Adams parameters 
    v = {} 
    m = {}
    beta_1, beta_2 =  0.9, 0.999
    alpha, epsilon = 0.01, 10**(-8)
    while (itr < max_it ):
        
        Z, Upsilon = F_tilde(Y, th, d_0, d, K, h)
        
        dJ = dJ_func(c, Y, th, d_0, d, K, h)
        #dJ_w, dJ_mu, dJ_W, dJ_b = dJ_func(c, Y, th, d_0, d, K, h)
        
        if (method=="gd"):
            #th = gradientDesent(K, th, dJ_w, dJ_mu, dJ_W, dJ_b, tau)
            th = gradientDesent(K, th, dJ, tau)
        
        elif (method=="adam"):
            j = itr
            
            dJ_w, dJ_mu, dJ_W, dJ_b = dJ_func(c, Y, th, d_0, d, K, h)
            
            for key in th:
                print(dJ)
            """
                m[key] = beta_1*m[key] + (1- beta_1)*g
                v[key] = beta_2*v[key] + (1 - beta_2)*(g*g)
                mhat = m[key]/(1 - beta_1**j)
                vhat = v[key]/(1 - beta_2**j)
                th[key] -= alpha*mhat/(np.square(vhat) + epsilon
            """
        else:
            print("No optimization method")
        
        err = J_func(Upsilon, c)  
        
        JJ[itr+1] = err
        
        itr += 1
        if(itr%50 == 0) and (print_it == True):
            print(itr,err)
        
    return JJ
        

import matplotlib.pyplot as plt
    
def main():
    K = 14
    h = 1/10
    d_0 = 2
    d = 4
    I = 20
    max_it = 1000
    tau = 0.5
                
    b = generate_synthetic_batches(I)
    
    #c = b["c"]
    #Y = b["Y"]
    c = scale(b["c"])
    Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    th = initialize_weights(d_0, d, K)
    JJ = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method="gd")
    it = np.arange(JJ.shape[0])
    plt.plot(it, JJ)
    plt.show()
    

#f()
#tau()
#layers()
main()
