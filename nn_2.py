#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:29:57 2020

@author: isakh
"""

import numpy as np
import matplotlib.pyplot as plt

from data import generate_synthetic_batches


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
    return ( (b - x)*alpha + (x - a)*beta)/(b - a), a, b, alpha, beta

def  invscale(x, a, b, alpha, beta):
    
    return ((x+alpha)*b - (x-beta)*a) / (beta-alpha)
    



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

def n(x):
    return np.linalg.norm(x)

def train(c, d, d_0, K, h, Y, th, tau=0.0005, max_it=60, print_it=False):
    # compute Zk
    err = np.inf
    tol = 0.01
    
    
    itr = 0
    Z, Upsilon = F_tilde(Y, th, d_0, d, K, h)
    JJ = np.zeros(max_it+1)
    err = J_func(Upsilon,c)
    
    JJ[0] = err
    
    while (itr < max_it ) and (err > tol):
        
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
        
        JJ[itr+1] = err
        
        itr += 1
        if(itr%50 == 0) and (print_it == True):
            print(itr,err)
        
    return JJ , th
        
def stocgradient(c, d, d_0, K, h, Y, th, tau, max_it , bsize):
    
    indexes = np.array(range(Y.shape[1]))
    
    JJ = np.array([])
    
    itr = 0
    
    while len(indexes) > 0 and itr < 500:
        print(itr,len(indexes))
        itr +=1
        if len(indexes) >= bsize:
            bsliceI = np.random.choice( indexes, bsize)
            Yslice = Y[:,bsliceI]
            cslice = c[bsliceI]
            
            dJJ, th = train(cslice, d, d_0, K, h, Yslice, th, tau, max_it)
            
            JJ = np.append(JJ,dJJ)
            
            indexes = np.delete(indexes,bsliceI)
            
        else:
            Yslice = Y[:,indexes]
            cslice = c[indexes]
            
            
            dJJ, th = train(cslice, d, d_0, K, h, Yslice, th, tau, max_it)
            
            JJ = np.append(JJ,dJJ)
            
            indexes = np.delete(indexes,indexes)
            
    return JJ, th
    
def main():
    K = 20
    h = 0.1
    I = 80
    max_it = 300
    tau = 0.1
    
    batches = import_batches()
    batch = batches[0]
    testbatch = batches[1]
    
    Y = batch["Y_q"]
    c,a,b,alfa,beta = scale(batch["c_q"])
    d_0 = Y.shape[0]
    d = d_0*2
    
    
    th = initialize_weights(d_0, d, K)
    #JJ, th = train(c, d, d_0, K, h, Y, th, tau, max_it)
    
    JJ, th = stocgradient(c, d, d_0, K, h, Y, th, tau, max_it , 100)
    
    #plt.plot(JJ)
    
    tY = testbatch["Y_q"]
    tc,a,b,alfa,beta = scale(batch["c_q"])
    
    z, yhat = F_tilde(tY, th, d_0, d, K, h)
    
    #y = invscale(yhat, a, b, alpha, beta) 
    
    plt.plot(yhat)
    plt.plot(tc)
    
    
    

    """            
    b = generate_synthetic_batches(I,"1sqr")
    
    
    
    Y = b["Y"]
    #Y,a,b,alfa,beta = scale(b["Y"])
    #c = b["c"]
    c,a,b,alfa,beta = scale(b["c"])
    
    d_0 = Y.shape[0]
    d = d_0*2
    
    
    th = initialize_weights(d_0, d, K)
    JJ, th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it)
    
    x = np.linspace(-2, 2, 200)
    x = np.reshape(x,(1,len(x)))
    #y = 1-np.cos(x)
    y = 1/2 *x**2
    z, yhat = F_tilde(x, th, d_0, d, K, h)
    yhat = invscale(yhat, a, b, alpha, beta)
    yhat = yhat.T
    
    plt.plot(x.T,y.T)
    plt.plot(x.T,yhat.T)
    """
    
    

#f()
#tau()
#layers()
main()
