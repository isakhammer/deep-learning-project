# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:12:59 2020

@author: Lisbeth
"""

import numpy as np
from numpy.linalg import norm as n
from copy import deepcopy as copy 
import sys
import matplotlib.pyplot as plt
import pickle

from data import *
from nn_2 import *

def train_two_body(pq):
    if pq == "p":
        func = "2sqr"
    elif pq == "q":
        func = "2norm-1"
    else:
        raise Exception("p or q")
    
    
    I = 8000
    K = 20
    h = 0.1
    sifts = 100
    
    qdata = generate_synthetic_batches(I, func=func)
    
    q =qdata["Y"]
    cq = qdata["c"]
    scq,invqc = scale(cq)
    
    parametersq = scale(cq,returnParameters = True)
    
    
    
    invq_file = open( pq + "_tb_inv.pkl", "wb")
    pickle.dump(parametersq, invq_file)
    invq_file.close()
    
    d_0 = q.shape[0]
    d = d_0*2
    
    Ihat = 40
    tau = 3/Ihat
    
    thq = initialize_weights(d_0, d, K)
    
    JJq, thq = stocgradient(scq, d, d_0, K, h, q, thq, tau, 1 , Ihat, sifts)
    
    plt.plot(JJq)
    plt.yscale("log")
    plt.show()
    
    
    thq_file = open(func + "_inv.pkl", "wb")
    pickle.dump(thq, thq_file)
    thq_file.close()

"""
def train_two_body():
    I = 8000
    K = 20
    h = 0.1
    sifts = 100
    
    pdata = generate_synthetic_batches(I,func = "2sqr")
    qdata = generate_synthetic_batches(I,func = "2norm-1")
    
    p = pdata["Y"]
    cp = pdata["c"]
    scp,invpc = scale(cp)
    parametersp = scale(cp,returnParameters = True)
    
    q =qdata["Y"]
    cq = qdata["c"]
    scq,invqc = scale(cq)
    
    parametersq = scale(cq,returnParameters = True)
    
    
    invp_file = open("tbinvp.pkl", "wb")
    pickle.dump(parametersp, invp_file)
    invp_file.close()
    
    invq_file = open("tbinvq.pkl", "wb")
    pickle.dump(parametersq, invq_file)
    invq_file.close()
    
    
    
    d_0 = p.shape[0]
    d = d_0*2
    
    Ihat = 40
    tau = 3/Ihat
    
    thp = initialize_weights(d_0, d, K)
    thq = initialize_weights(d_0, d, K)
    
    JJp, thp = stocgradient(scp, d, d_0, K, h, p, thp, tau, 1 , Ihat, sifts)
    JJq, thq = stocgradient(scq, d, d_0, K, h, q, thq, tau, 1 , Ihat, sifts)
    
    plt.plot(JJp)
    plt.plot(JJq)
    plt.yscale("log")
    plt.show()
    
    thp_file = open("tbpw.pkl", "wb")
    pickle.dump(thp, thp_file)
    thp_file.close()
    
    thq_file = open("tbqw.pkl", "wb")
    pickle.dump(thq, thq_file)
    thq_file.close()
    
 """   
    
def test_two_body():
    
    numData = 2000
    
    K = 20
    h = 0.1
    d_0 = 2
    d = 4
    
    
    
    x = np.linspace(-2,2,numData)
    
    p = np.array([x,-1/2*x])
    
    q = np.array([-1/3*x,x])
    
    pc = 0.5*p[0]**2 + 0.5*p[1]**2
    pc = pc[:, np.newaxis]
    
    qc = -1/np.sqrt(q[0]**2 + q[1]**2)
    qc = qc.T
    qc = qc[:, np.newaxis]
    
    
    
    pp_file = open("tbinvp.pkl", "rb")
    pinvp = pickle.load(pp_file)
    pp_file.close()
    
    qp_file = open("tbinvq.pkl", "rb")
    qinvp = pickle.load(qp_file)
    qp_file.close()
    
    pw_file = open("tbpw.pkl", "rb")
    thp = pickle.load(pw_file)
    pw_file.close()
    
    qw_file = open("tbqw.pkl", "rb")
    thq = pickle.load(qw_file)
    qw_file.close()
    
    
    zp, yhatp = F_tilde(p, thp, d_0, d, K, h)
    zq, yhatq = F_tilde(q, thq, d_0, d, K, h)
    
    yp = invscaleparameter(yhatp, pinvp[0], pinvp[1], pinvp[2], pinvp[3])
    yq = invscaleparameter(yhatq, qinvp[0], qinvp[1], qinvp[2], qinvp[3])
    
    
    plt.plot(yq)
    #plt.plot(qc)
    plt.show()
    plt.plot(yp)
    plt.plot(pc)
    plt.show()
    
    
    
    
    
    
    
def train_nlp(pq):
    
    if pq == "p":
        func = "1sqr"
    elif pq == "q":
        func = "1cos"
    else:
        raise Exception("p or q")
    
    
    I = 8000
    K = 20
    h = 0.1
    sifts = 300
    Ihat = 40
    tau = 3/Ihat
    
    data = generate_synthetic_batches(I, func)
    
    Y =data["Y"]
    c = data["c"]
    sc , invc = scale(c)
    sparameters = scale(c,returnParameters = True)
    
    inv_file = open( pq + "_nlp_inv.pkl", "wb")
    pickle.dump(sparameters, inv_file)
    inv_file.close()
    
    d_0 = Y.shape[0]
    d = d_0*2
    
    
    
    th = initialize_weights(d_0, d, K)
    
    JJ, th = stocgradient(sc, d, d_0, K, h, Y, th, tau, 1 , Ihat, sifts)
    
    plt.plot(JJ)
    plt.yscale("log")
    plt.show()
    
    
    th_file = open(pq + "_nlp_w.pkl", "wb")
    pickle.dump(th, th_file)
    th_file.close()
    
    
    
def test_nlp(pq):
    
    numData = 20
    
    K = 20
    h = 0.1
    d_0 = 1
    d = 2
    
    if pq == "p":
        Y  = np.linspace(-2,2,numData)
        Y = Y[:,np.newaxis].T
        c = 1/2*Y**2
        c = c.T
        
    elif pq == "q":
        Y = np.linspace(-np.pi/3,np.pi/3,numData)
        Y = Y[:,np.newaxis].T
        c = 1-np.cos(Y)
        c = c.T
    else:
        raise Exception("p or q")
        
    
    
    
    inv_file = open( pq + "_nlp_inv.pkl", "rb")
    inv = pickle.load(inv_file)
    inv_file.close()
    
    w_file = open(pq + "_nlp_w.pkl", "rb")
    th = pickle.load(w_file)
    w_file.close()
    print(Y)
    z, yhat = F_tilde(Y, th, d_0, d, K, h)
    
    y = invscaleparameter(yhat, inv[0], inv[1], inv[2], inv[3])
    print(y)
    
    plt.plot(Y.T,y)
    plt.plot(Y.T,c)
    plt.show()


def test_euler():
    
    K = 20
    hF = 0.1
    d_0 = 1
    d = d_0*2
    
    T = 6
    dt = 1e-3
    N = int(T/dt)
    
    invp_file = open("p_nlp_inv.pkl", "rb")
    invp = pickle.load(invp_file)
    invp_file.close()
    
    invq_file = open("q_nlp_inv.pkl", "rb")
    invq = pickle.load(invq_file)
    invq_file.close()
    
    wp_file = open("p_nlp_w.pkl", "rb")
    thp = pickle.load(wp_file)
    wp_file.close()

    wq_file = open("q_nlp_w.pkl", "rb")
    thq = pickle.load(wq_file)
    wq_file.close()
    
    
    p0 = np.array([1])[:,np.newaxis]
    q0 = np.array([0])[:,np.newaxis]
    
    
    p,q = s_euler(p0, q0, thp, thq, hF, K, N, T, invp, invq)
    
    #p = invscaleparameter_no_shift(p, invp[0], invp[1], invp[2], invp[3])
    #q = invscaleparameter_no_shift(q, invq[0], invq[1], invq[2], invq[3])
    
    plt.plot(p)
    plt.plot(q)
    #plt.plot(p+q)
    plt.show()
    
    """
    zp, Tps = F_tilde(p[:,np.newaxis].T, thp, d_0, d, K, hF)
    zq, Vqs = F_tilde(q[:,np.newaxis].T, thq, d_0, d, K, hF)
    
    Tp = invscaleparameter(Tps, invp[0], invp[1], invp[2], invp[3])
    Vq = invscaleparameter(Vqs, invq[0], invq[1], invq[2], invq[3])
    """
    Tp = 1/2*p**2
    Vq = 1-np.cos(q)
    
    
    plt.plot(np.reshape(Tp,len(Tp)))
    plt.plot(np.reshape(Vq,len(Vq)))
    plt.plot(Tp+Vq)
    plt.show()
    
    
    
    
    
#test_two_body()
#train_two_body()

#train_nlp("p")


#train_nlp("q")
    
#test_nlp("p")
#test_nlp("q")

test_euler()
    





    
    