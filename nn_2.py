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


def scale(x, alpha=0, beta=1, returnParameters = False):
    
    a = np.min(x)
    b = np.max(x)
    
    if returnParameters:
        
        return alpha, beta, a, b
        
    else:
        
        def  invscale(x):
            return ((x+alpha)*b - (x-beta)*a) / (beta-alpha)
        
        return ( (b - x)*alpha + (x - a)*beta)/(b - a), invscale
    

    
def invscaleparameter(x, alpha, beta, a, b):
    return ((x+alpha)*b - (x-beta)*a) / (beta-alpha)

def invscaleparameter_no_shift(x, alpha, beta, a, b):
    return x*(b-a)/(beta-alpha)

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


def adam_algebra(th, dJ, v, m, key, j, alpha =10**(-5)):
        beta_1, beta_2 =  0.9, 0.999
        epsilon =  10**(-8)
    
    
        g = dJ[key] 
        m[key] = beta_1*m[key] + (1- beta_1)*g
        v[key] = beta_2*v[key] + (1 - beta_2)*(g*g)
        mhat = m[key]/(1 - beta_1**(j+1))
        vhat = v[key]/(1 - beta_2**(j+1))
        th[key] -= alpha*mhat/(np.square(vhat) + epsilon)
        #print("hat",  vhat, v[key], mhat, m[key], j+1 )
        return th, v, m
    


def train(c, d, d_0, K, h, Y, th, tau=0.0005, max_it=60, print_it=True, method="gd", alpha =7.5*10**(-5)):
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
    
    
    
    
    while (itr < max_it ):
        
        Z, Upsilon = F_tilde(Y, th, d_0, d, K, h)
        
        
        
        if (method=="gd"):
            dJ = dJ_func(c, Y, th, d_0, d, K, h)
            th = gradientDesent(K, th, dJ, tau)
        
        elif (method=="adam"):
            j = itr
            
            dJ = dJ_func(c, Y, th, d_0, d, K, h)
            
            th, v, m = adam_algebra(th, dJ, v, m, "mu", j, alpha)
            th, v, m = adam_algebra(th, dJ, v, m, "w", j, alpha)
            th, v, m = adam_algebra(th, dJ, v, m, "W", j, alpha)
            th, v, m = adam_algebra(th, dJ, v, m, "b", j, alpha)
            
        else:
            print("No optimization method")
        
        err = J_func(Upsilon, c)  
        
        JJ[itr+1] = err
        
        itr += 1
        
        if(itr%600 == 0) and (print_it == True):
            print(itr,err)
        
    return JJ , th
        
def stocgradient(c, d, d_0, K, h, Y, th, tau, max_it , bsize, sifts = 100, save = False, savefile = ""):
    
    JJ = np.array([])
    I = Y.shape[1]
    totitr = int(I/bsize)
    for siftnum in range(sifts):
        print(siftnum)
        
        indexes = np.array(range(I))
        
        np.random.shuffle(indexes)
        
        itr = 0
        
        Z, Upsilon = F_tilde(Y, th, d_0, d, K, h)
        err = J_func(Upsilon, c)
        JJ = np.append(JJ, err)
        
        if save and siftnum%100 == 0:
            th_file = open(savefile, "wb")
            pickle.dump(th, th_file)
            th_file.close()
        
        while len(indexes) > 0:
            """
            if (itr % 50 == 0):
                print(siftnum,itr,totitr)
            itr +=1
            """
            if len(indexes) >= bsize:
                #bsliceI = np.random.choice( indexes, bsize, False)
                bsliceI = indexes[:bsize]
                Yslice = Y[:,bsliceI]
                cslice = c[bsliceI]
                
                dJJ, th = train(cslice, d, d_0, K, h, Yslice, th, tau, max_it)
                
                #JJ = np.append(JJ,dJJ)
                
                indexes = indexes[bsize:]
                
                
            else:
                Yslice = Y[:,indexes]
                cslice = c[indexes]
                
                
                dJJ, th = train(cslice, d, d_0, K, h, Yslice, th, tau, max_it)
                
                #JJ = np.append(JJ,dJJ)
                
                indexes = []
            
    return JJ, th


def variablestocgradient(c, d, d_0, K, h, Y, th, tau, max_it, sifts):
    
    bsizes = np.array([10, 20, 40, 80, 160, 360])
    
    JJ = np.array([])
    
    for bsize in bsizes:
        
        dJJ, th = stocgradient(c, d, d_0, K, h, Y, th, tau, int(bsize/10) , bsize, sifts)
        
        JJ = np.append(JJ,dJJ)
    
    
    return JJ, th

def dF_tilde_y(y, h, th, d_0, d, K):
    
    Z, Upsilon = F_tilde(y, th, d_0, d, K, h)
    
    dz =  np.identity(d)[:,:d_0]    
    for k in range(0,K):
        dz =  h* sigma(th["W"][k]@ Z[k] +  th["W"][k], derivative = True)@(th["W"][k] @dz) + dz     
    dUpsilon = eta(Z[K].T @ th["w"]  + th["mu"], derivative=True )  @ (th["w"].T@dz)
    #print("zk", Z[K])
    return dUpsilon 

def dF_tilde_y2(y, h, th, d_0, d, K):
    
    Z, Upsilon = F_tilde(y, th, d_0, d, K, h)
    
    
    dGZK = eta(th["w"][:d_0].T@Z[K][:d_0]+ th["mu"], derivative = True) * th["w"][:d_0]
    
    A = dGZK
    
    for k in range(K,0,-1):
        A = A + th["W"][k-1][:d_0,:d_0]@(h*sigma((th["W"][k-1][:d_0,:d_0]@Z[k-1][:d_0] + th["b"][k-1][:d_0]), derivative = True)*A)
   
    """
    dGZK = eta(th["w"].T@Z[K]+ th["mu"], derivative = True) * th["w"]
    
    A = dGZK
    for k in range(K,0,-1):
        A = A + th["W"][k-1]@(h*sigma((th["W"][k-1]@Z[k-1] + th["b"][k-1]), derivative = True)*A)
    """
    
    return A

def s_stormer(p0, q0, thp, thq, hF, K, N, T, invp, invq):
    
    h = T/N
    
    d_0 = p0.shape[0]
    d = d_0*2
    
    p = np.zeros((N+1,1,1))
    p[0] = p0
    
    q = np.zeros((N+1,1,1))
    q[0] = q0
    
    for n in range(N):
        
        # 1
        """
        dV = np.sin(q[n])
        phat = p[n] - h/2*dV
        dT = phat
        q[n+1] = q[n] + h*dT
        dV = np.sin(q[n+1])
        p[n+1] = phat - h/2*dV
        
        """
        dVs = dF_tilde_y(q[n], hF, thq, d_0, d, K)
        dV = invscaleparameter_no_shift(dVs, invq[0], invq[1], invq[2], invq[3])
        p_hat = p[n] - (h/2)*dV
        
        # 2
        dTs = dF_tilde_y(p_hat, hF, thp, d_0, d, K)
        dT = invscaleparameter_no_shift(dTs, invp[0], invp[1], invp[2], invp[3])
        q[n+1] = q[n] + h*dT        
        
        # 3
        dVs = dF_tilde_y( q[n+1], hF, thq, d_0, d, K)
        dV1 = invscaleparameter_no_shift(dVs, invq[0], invq[1], invq[2], invq[3])
        p[n+1] = p_hat - (h/2)*dV1
        
        
    p = np.reshape(p,N+1)
    q = np.reshape(q,N+1)
    return p,q
    
def s_euler(p0, q0, thp, thq, hF, K, N, T, invp, invq):
    
    
    h = T/N
    
    d_0 = p0.shape[0]
    d = d_0*2
    
    p = np.zeros((N+1,1,1))
    p[0] = p0
    
    q = np.zeros((N+1,1,1))
    q[0] = q0
    
    for n in range(N):
        #print(n,N)
        """
        dT = p[n]
        dV = np.sin(q[n])
        """
        
        dTs = dF_tilde_y2(p[n], hF, thp, d_0, d, K)
        dT = invscaleparameter_no_shift(dTs, invp[0], invp[1], invp[2], invp[3])
        dVs = dF_tilde_y2(q[n], hF, thq, d_0, d, K)
        dV = invscaleparameter_no_shift(dVs, invq[0], invq[1], invq[2], invq[3])
        
        """
        dTs = dF_tilde_y(p[n], hF, thp, d_0, d, K)
        dT = invscaleparameter_no_shift(dTs, invp[0], invp[1], invp[2], invp[3])
        dVs = dF_tilde_y(q[n], hF, thq, d_0, d, K)
        dV = invscaleparameter_no_shift(dVs, invq[0], invq[1], invq[2], invq[3])
        """
        q[n+1] = q[n] + h*dT[:d_0]
        p[n+1] = p[n] - h*dV[:d_0]
    
    p = np.reshape(p,N+1)
    q = np.reshape(q,N+1)
    return p,q
    

def main_magnus():
    K = 20
    h = 0.1
    #bsize = 1024
    bsize = 40
    max_it = 1
    sifts = 300
    #tau = 0.1
    tau = 2/bsize
    
    
    batches = import_batches()
    batch1 = batches[0]
    antB = 3
    testbatch = batches[antB-1]
    
    
    
    Y = batch1["Y_q"]
    #c,a,b,alfa,beta = scale(batch1["c_q"])
    d_0 = Y.shape[0]
    d = d_0*2
    
    I = Y.shape[1]
    #tau = 2/I
    
    Ihat = 40
    tau = 2/Ihat
    
    
    
    
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
        #ci,invi = scale(batch["c_q"])
        #bigbatch["c"] = np.append(bigbatch["c"],ci)
        
    Y = bigbatch["Y"]
    c,inv = scale(bigbatch["c"][:,np.newaxis])
    
    
    """
    s_parameters = scale(bigbatch["c"][:,np.newaxis],returnParameters = True)
    
    invP_file = open("invP.pkl", "wb")
    pickle.dump(s_parameters, invP_file)
    invP_file.close()
    """
    #c = bigbatch["c"][:,np.newaxis]
    
    """
    Y = Y[:,:3000]
    c = c[:3000,:]
    """
    
    JJ, th = stocgradient(c, d, d_0, K, h, Y, th, tau, 1 , bsize, sifts)
    #JJ, th = variablestocgradient(c, d, d_0, K, h, Y, th, tau, 1 , max_it)
    #JJ, th = train(c, d, d_0, K, h, Y, th, tau, sifts)    
    ####
    
    
    ####
    """
    cycles = 1
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
    tc,invscale = scale(testbatch["c_q"])
    
    z, yhat = F_tilde(tY, th, d_0, d, K, h)
    
    y = inv(yhat)
    ic = invscale(tc)
    
    plt.plot(y)
    plt.plot(ic)
    plt.show()
    
    th_file = open("weights.pkl", "wb")
    pickle.dump(th, th_file)
    th_file.close()

    
    
    

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
    
    #th_file = open("weights.pkl", "rb")
    th_file = open("weights_I40_300.pkl", "rb")
    th = pickle.load(th_file)
    th_file.close()
    
    
    p_file = open("invP.pkl", "rb")
    alpha,beta,a,b = pickle.load(p_file)
    p_file.close
    
    
    
    batches = import_batches()
    batch1 = batches[0]
    antB = 49
    
    
    Y = batch1["Y_q"]
    d_0 = Y.shape[0]
    d = d_0*2
    
    for i in range(antB):
        print(i)
        testbatch = batches[i]
        
        
    
        tY = testbatch["Y_q"]
        tc,invt = scale(testbatch["c_q"])
        
        z, yhat = F_tilde(tY, th, d_0, d, K, h)
        
        #ic = invt(tc)
        
        y = invscaleparameter(yhat, alpha, beta, a, b)
        ic = invt(tc)
        
        
        
        plt.plot(y)
        plt.plot(ic)
        plt.show()
    
    

def main_isak():
    K = 20
    h = 0.1
    I = 8000
    #tau = 0.1
    tau = 2/I
    max_it = 10000

    b = generate_synthetic_batches(I,"2norm-1")
    #b = generate_synthetic_batches(I)

    Y = b["Y"]
    #Y,a,b,alfa,beta = scale(b["Y"])
    #c = b["c"]
    c, inv = scale(b["c"])

    d_0 = Y.shape[0]
    d = d_0*2


    th = initialize_weights(d_0, d, K)
    #JJ, th = train(c, d, d_0, K, h, Y, th, tau, max_it, method="gd")
    #JJ, th = train(c, d, d_0, K, h, Y, th, tau, max_it, method="adam")
    """
    x = np.linspace(-2, 2, I)
    x = np.reshape(x,(1,len(x)))
    #y = 1-np.cos(x)
    y = 1/2 *x**2
    z, yhat = F_tilde(x, th, d_0, d, K, h)
    yhat = inv(yhat)
    yhat = yhat.T

    plt.plot(x.T,y.T)
    plt.plot(x.T,yhat.T)
    plt.show()
    """
    #plt.plot(JJ)
    #plt.show()
    
    plt.plot(JJ)
    plt.yscale("log")
    plt.show()
    
    

#main_magnus()
#test_weights()
#main_isak()
#plt.show()
