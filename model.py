#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:29:57 2020

@author: isakh
"""

import numpy as np
from numpy.linalg import norm as n
from copy import deepcopy as copy 
import sys
import matplotlib.pyplot as plt
import pickle

from data import *


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
    return ( (b - x)*alpha + (x - a)*beta)/(b - a), a, b, alpha, beta

def  invscale(x, a, b, alpha, beta):
    
    return ((x+alpha)*b - (x-beta)*a) / (beta-alpha)
    


def gradientDesent(K, th, dJ, tau):
    
    th["mu"] = th["mu"] - tau*dJ["mu"]
    th["w"] = th["w"] - tau*dJ["w"]
    
    th["W"] = th["W"] -  tau*dJ["W"]
    th["b"] = th["b"] -  tau*dJ["b"]
    
    return th


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

    return dJ




def train(c, d, d_0, K, h, Y, th, tau=0.0005, max_it=60, print_it=False, method="gd"):
    # compute Zk
    err = np.inf
    tol = 0.01
    
    
    itr = 0
    Z, Upsilon = F_tilde(Y, th, d_0, d, K, h)
    JJ = np.zeros(max_it+1)
    err = J_func(Upsilon,c)
    
    JJ[0] = err
    
    # Adam parameters 
    m = {}
    m["mu"] = np.zeros(th["mu"].shape)
    m["w"] = np.zeros(th["w"].shape)
    m["W"] = np.zeros(th["W"].shape)
    m["b"] = np.zeros(th["b"].shape)
    v = copy(m)
    
    beta_1, beta_2 =  0.9, 0.999
    alpha, epsilon = 0.01, 10**(-8)
    
    def adam_algebra(th, dJ, v, m, key, j):
        g = dJ[key] 
        m[key] = beta_1*m[key] + (1- beta_1)*g
        v[key] = beta_2*v[key] + (1 - beta_2)*(g*g)
        mhat = m[key]/(1 - beta_1**(j+1))
        vhat = v[key]/(1 - beta_2**(j+1))
        th[key] -= alpha*mhat/(np.square(vhat) + epsilon)
        #print("hat",  vhat, v[key], mhat, m[key], j+1 )
        return th, v, m
    
    
    while (itr < max_it ):
        
        Z, Upsilon = F_tilde(Y, th, d_0, d, K, h)
        
        
        
        if (method=="gd"):
            dJ = dJ_func(c, Y, th, d_0, d, K, h)
            th = gradientDesent(K, th, dJ, tau)
        
        elif (method=="adam"):
            j = itr
            
            dJ = dJ_func(c, Y, th, d_0, d, K, h)
            
            th, v, m = adam_algebra(th, dJ, v, m, "mu", j)
            th, v, m = adam_algebra(th, dJ, v, m, "w", j)
            th, v, m = adam_algebra(th, dJ, v, m, "W", j)
            th, v, m = adam_algebra(th, dJ, v, m, "b", j)
            
        else:
            print("No optimization method")
        
        err = J_func(Upsilon, c)  
        
        JJ[itr+1] = err
        
        itr += 1
        if(itr%50 == 0) and (print_it == True):
            print(itr,err)
        
    return JJ , th
        
def stocgradient(c, d, d_0, K, h, Y, th, tau, max_it , bsize, sifts = 1):
    
    JJ = np.array([])
    
    for siftnum in range(sifts):
        #print(siftnum)
        I = Y.shape[1]
        totitr = int(I/bsize)
        
        indexes = np.array(range(I))
        
        itr = 0
        
        while len(indexes) > 0:
            print(siftnum,itr,totitr)
            itr +=1
            if len(indexes) >= bsize:
                bsliceI = np.random.choice( indexes, bsize)
                Yslice = Y[:,bsliceI]
                cslice = c[bsliceI]
                
                dJJ, th = train(cslice, d, d_0, K, h, Yslice, th, tau, max_it)
                
                JJ = np.append(JJ,dJJ)
                
                si = np.zeros(bsize).astype(int)
                
                for i in range(bsize):
                    si[i] = np.where(indexes == bsliceI[i])[0]
                
                indexes = np.delete(indexes,si)
                #indexes = np.delete(indexes,np.where(indexes == bsliceI))
                
                
            else:
                Yslice = Y[:,indexes]
                cslice = c[indexes]
                
                
                dJJ, th = train(cslice, d, d_0, K, h, Yslice, th, tau, max_it)
                
                JJ = np.append(JJ,dJJ)
                
                indexes = []
            
    return JJ, th
    
def dF_tilde_y(y, h, th, d_0, d, K ):
    
    Z, Upsilon = F_tilde(y, th, d_0, d, K, h)
    dz =  np.identity(d)[:,:d_0]    
    for k in range(0,K):
        dz =  h* sigma(th["W"][k]@ Z[k] +  th["W"][k], derivative = True)@(th["W"][k] @dz) + dz     
    dUpsilon = eta(Z[K].T @ th["w"]  + th["mu"] )  @ (th["w"].T@dz)
    return dUpsilon 

def main_magnus():
    K = 20
    h = 0.1
    I = 80
    max_it = 300
    tau = 0.1
    
    batches = import_batches()
    batch1 = batches[0]
    antB = 10
    testbatch = batches[antB-1]
    
    cycles = 1
    
    Y = batch1["Y_q"]
    #c,a,b,alfa,beta = scale(batch1["c_q"])
    d_0 = Y.shape[0]
    d = d_0*2
    
    
    th = initialize_weights(d_0, d, K)
    #JJ, th = train(c, d, d_0, K, h, Y, th, tau, max_it)
    JJ = np.array([])
    
    
    ####
    
    bigbatch = {}
    bigbatch["Y"] = np.array([[],[],[]])
    bigbatch["c"] = np.array([])
    
    for i in range(antB):
        batch = batches[i]
        bigbatch["Y"] = np.append(bigbatch["Y"],batch["Y_q"],1)
        bigbatch["c"] = np.append(bigbatch["c"],batch["c_q"])
        
    Y = bigbatch["Y"]
    c,a,b,alfa,beta = scale(bigbatch["c"][:,np.newaxis])
    
        
    JJ, th = stocgradient(c, d, d_0, K, h, Y, th, tau, 1 , 160, max_it)
        
    ####
    
    
    ####
    """
    for cycle in range(cycles):
        for i in range(antB):
            print(cycle,i)
            batch = batches[i]
            Y = batch["Y_q"]
            c,a,b,alfa,beta = scale(batch["c_q"])
            
            dJJ, th = stocgradient(c, d, d_0, K, h, Y, th, tau, max_it , 160)
            #dJJ, th = train(c, d, d_0, K, h, Y, th, tau, max_it)
            JJ = np.append(JJ,dJJ)
    """
    ####
        
    plt.plot(JJ)
    plt.yscale("log")
    plt.show()
    
    tY = testbatch["Y_q"]
    tc,a,b,alfa,beta = scale(testbatch["c_q"])
    
    z, yhat = F_tilde(tY, th, d_0, d, K, h)
    
    #y = invscale(yhat, a, b, alpha, beta) 
    
    plt.plot(yhat)
    plt.plot(tc)
    plt.show()
    
    th_file = open("weights.pkl", "wb")
    pickle.dump(th, th_file)
    th_file.close()

    #a_file = open("data.pkl", "rb")
    #output = pickle.load(a_file)
    #print(output)
    
    
    

    """            
    b = generate_synthetic_batches(I,"1sqr")
    
    Y = b["Y"]
    #Y,a,b,alfa,beta = scale(b["Y"])
    #c = b["c"]
    c,a,b,alpha,beta = scale(b["c"])
    
    d_0 = Y.shape[0]
    d = d_0*2
    
    
    th = initialize_weights(d_0, d, K)
    JJ, th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method=method)
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

def test_weights():
    
    K = 20
    h = 0.1
    tau = 0.1
    
    th_file = open("weights.pkl", "rb")
    th = pickle.load(th_file)
    
    batches = import_batches()
    batch1 = batches[0]
    antB = 3
    testbatch = batches[10]
    
    Y = batch1["Y_q"]
    c,a,b,alfa,beta = scale(batch1["c_q"])
    d_0 = Y.shape[0]
    d = d_0*2
    
    tY = testbatch["Y_q"]
    tc,a,b,alfa,beta = scale(testbatch["c_q"])
    
    z, yhat = F_tilde(tY, th, d_0, d, K, h)
    
    plt.plot(yhat)
    plt.plot(tc)
    plt.show()
    
    

def main_isak():
    K = 20
    h = 0.1
    I = 130
    tau = 0.25
    max_it = 3000

    b = generate_synthetic_batches(I,"2sqr")

    Y = b["Y"]
    #Y,a,b,alfa,beta = scale(b["Y"])
    #c = b["c"]
    c,a,b,alfa,beta = scale(b["c"])

    d_0 = Y.shape[0]
    d = d_0*2


    th = initialize_weights(d_0, d, K)
    #JJ = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method="gd")
    JJ, th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method="gd")
    y = Y[:,0, np.newaxis]
    dUps = dF_tilde_y(y, h, th, d_0, d, K )
    it = np.arange(JJ.shape[0])
    plt.plot(it, JJ)

#main_magnus()
# main_isak()
#plt.show()
#test_weights()