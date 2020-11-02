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

def train_two_body(pq, continue_training = False):
    
    if pq == "p":
        func = "2sqr"
    elif pq == "q":
        func = "2norm-1"
    else:
        raise Exception("p or q")
    
    
    I = 8000
    K = 20
    h = 0.1
    sifts = 1000
    Ihat = 400
    tau = 2/Ihat
    
    qdata = generate_synthetic_batches(I, func=func)
    
    
    q =qdata["Y"]
    cq = qdata["c"]
    scq,invqc = scale(cq)
    
    parametersq = scale(cq, returnParameters = True)
    
    
    
    invq_file = open( pq + "_tb_inv.pkl", "wb")
    pickle.dump(parametersq, invq_file)
    invq_file.close()
    
    d_0 = q.shape[0]
    d = d_0*2
    
    
    if continue_training:
        qw_file = open( pq + "_tb_w.pkl", "rb")
        thq = pickle.load(qw_file)
        qw_file.close()
        
    else:
        thq = initialize_weights(d_0, d, K)
        
    
    JJq, thq = stocgradient(scq, d, d_0, K, h, q, thq, tau, 1 , Ihat, sifts)
        
    
    plt.plot(JJq)
    plt.yscale("log")
    plt.show()
    
    
    thq_file = open(pq + "_tb_w.pkl", "wb")
    pickle.dump(thq, thq_file)
    thq_file.close()
 
    
def test_two_body():
    
    numData = 2000
    
    K = 20
    h = 0.1
    d_0 = 2
    d = 4
    
    x1 = np.linspace(-2,2,numData)
    
    x2hat1 = np.linspace(-2,-1/4,int(numData/2))
    x2hat2 = np.linspace(1/4,2,int(numData/2))
    x2 = np.append(x2hat1,x2hat2)
    
    p = np.array([x1,-1/2*x1])  
    q = np.array([x2,-x2])
    
    pc = 0.5*p[0]**2 + 0.5*p[1]**2
    pc = pc[:, np.newaxis]
    
    qc = -1/np.sqrt(q[0]**2 + q[1]**2)
    qc = qc.T
    qc = qc[:, np.newaxis]
    
    pp_file = open("p_tb_inv.pkl", "rb")
    pinvp = pickle.load(pp_file)
    pp_file.close()
    
    qp_file = open("q_tb_inv.pkl", "rb")
    qinvp = pickle.load(qp_file)
    qp_file.close()
    
    pw_file = open("p_tb_w.pkl", "rb")
    thp = pickle.load(pw_file)
    pw_file.close()
    
    qw_file = open("q_tb_w.pkl", "rb")
    thq = pickle.load(qw_file)
    qw_file.close()
    
    
    zp, yhatp = F_tilde(p, thp, d_0, d, K, h)
    zq, yhatq = F_tilde(q, thq, d_0, d, K, h)
    
    yp = invscaleparameter(yhatp, pinvp[0], pinvp[1], pinvp[2], pinvp[3])
    yq = invscaleparameter(yhatq, qinvp[0], qinvp[1], qinvp[2], qinvp[3])
    
    
    plt.plot(yq)
    plt.plot(qc)
    plt.show()
    plt.plot(yp)
    plt.plot(pc)
    plt.show()
    


def model_two_body():
    
    K = 20
    hF = 0.1
    d_0 = 1
    d = d_0*2
    
    T = 3
    dt = 1e-4
    N = int(T/dt)
    
    pp_file = open("p_tb_inv.pkl", "rb")
    invp = pickle.load(pp_file)
    pp_file.close()
    
    qp_file = open("q_tb_inv.pkl", "rb")
    invq = pickle.load(qp_file)
    qp_file.close()
    
    pw_file = open("p_tb_w.pkl", "rb")
    thp = pickle.load(pw_file)
    pw_file.close()
    
    qw_file = open("q_tb_w.pkl", "rb")
    thq = pickle.load(qw_file)
    qw_file.close()
    
    
    p0 = np.array([0.8,0])[:,np.newaxis]
    q0 = np.array([0,-1.2])[:,np.newaxis]
    
    
    def dT(p):
        derr = np.array([p[0],p[1]])
        return derr
    
    def dV(q):
        derr = np.array([q[0]/(q[0]**2+q[1]**2)**(3/2), q[1]/(q[0]**2+q[1]**2)**(3/2)])
        return derr
    
    #p,q = s_euler(p0, q0, thp, thq, hF, K, N, T, invp, invq)
    p,q = stormer_verlet(p0, q0, thp, thq, hF, K, N, T, invp, invq)
    pa,qa = stormer_verlet_analytical(p0, q0, N, T,  dT, dV)
    
    print(np.amax(pa))
    
    theta = np.linspace(0, 2*np.pi, 100)

    r = 1/4

    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)
    
    plt.plot(x1,x2, label = "radius")
    
    plt.plot(q[:,0],q[:,1],label="q")
    plt.plot(qa[:,0],qa[:,1],label="qa")
    plt.legend()
    plt.show()
    
    Tpa = 1/2*pa[:,0]**2 + 1/2*pa[:,1]**2
    Vqa = -1/np.sqrt(qa[:,0]**2 + qa[:,1]**2)
    
    plt.plot(Tpa, label="Ta")
    plt.plot(Vqa,  label ="Va")
    plt.plot(Tpa+Vqa, label ="Ta + Va")
    plt.legend()
    plt.show()
    
    Tp = 1/2*p[:,0]**2 + 1/2*p[:,1]**2
    Vq = -1/np.sqrt(q[:,0]**2 + q[:,1]**2)
    
    plt.plot(Tp, label="T")
    plt.plot(Vq,  label ="V")
    plt.plot(Tp+Vq, label ="T + V")
    plt.legend()
    plt.show()
    
    
    """
    plt.plot(p, label="p")
    plt.plot(q, label="q")
    plt.legend()
    plt.show()
    plt.plot(pa, label="pa")
    plt.plot(qa, label="qa")
    plt.legend()
    plt.show()
    
    Tp = 1/2*p**2
    Vq = 1-np.cos(q)
    
    Tpa = 1/2*pa**2
    Vqa = 1-np.cos(qa)
    
    
    plt.plot(np.reshape(Tp,len(Tp)), label="T")
    plt.plot(np.reshape(Vq,len(Vq)),  label ="V")
    plt.plot(Tp+Vq, label ="T + V")
    plt.legend()
    plt.show()
    
    """
    
    
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
    sifts = 2400
    Ihat = 320
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
    
    numData = 2000
    
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
    z, yhat = F_tilde(Y, th, d_0, d, K, h)
    
    y = invscaleparameter(yhat, inv[0], inv[1], inv[2], inv[3])
    
    plt.plot(Y.T,y, label ="y")
    plt.plot(Y.T,c, label ="c")
    plt.legend()
    plt.show()


def test_nlp_grad(pq):
    
    numData = 2000
    
    K = 20
    h = 0.1
    d_0 = 1
    d = 2
    
    if pq == "p":
        Y  = np.linspace(-2,2,numData)
        Y = Y[:,np.newaxis].T
        c = Y
        c = c.T
        
    elif pq == "q":
        Y = np.linspace(-np.pi/3,np.pi/3,numData)
        Y = Y[:,np.newaxis].T
        c = np.sin(Y)
        c = c.T
    else:
        raise Exception("p or q")
    
    
    inv_file = open( pq + "_nlp_inv.pkl", "rb")
    inv = pickle.load(inv_file)
    inv_file.close()
    
    w_file = open(pq + "_nlp_w.pkl", "rb")
    th = pickle.load(w_file)
    w_file.close()
    
    """
    yhat = np.zeros(numData)
    
    for i in range(numData):
        yhat[i] = dF_tilde_y2(Y.T[i,np.newaxis], h, th, d_0, d, K)
        
    y = invscaleparameter_no_shift(yhat, inv[0], inv[1], inv[2], inv[3])
    
    """
    yhat = dF_tilde_y2(Y, h, th, d_0, d, K)
    y = invscaleparameter_no_shift(yhat, inv[0], inv[1], inv[2], inv[3])
    
    
    plt.plot(y.T, label ="y")
    plt.plot(c, label ="c")
    #plt.axhline(y = 0)
    plt.legend()
    plt.show()


def model_nlp():
    
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
    
    
    p0 = np.array([0.5])[:,np.newaxis]
    q0 = np.array([0])[:,np.newaxis]
    
    
    def dT(p):
        return p
    
    def dV(q):
        return np.sin(q)
    
    #p,q = s_euler(p0, q0, thp, thq, hF, K, N, T, invp, invq)
    p,q = stormer_verlet(p0, q0, thp, thq, hF, K, N, T, invp, invq)
    pa,qa = stormer_verlet_analytical(p0, q0, N, T,  dT, dV)
    
    
    plt.plot(p, label="p")
    plt.plot(q, label="q")
    plt.legend()
    plt.show()
    plt.plot(pa, label="pa")
    plt.plot(qa, label="qa")
    plt.legend()
    plt.show()
    
    Tp = 1/2*p**2
    Vq = 1-np.cos(q)
    
    Tpa = 1/2*pa**2
    Vqa = 1-np.cos(qa)
    
    
    plt.plot(np.reshape(Tp,len(Tp)), label="T")
    plt.plot(np.reshape(Vq,len(Vq)),  label ="V")
    plt.plot(Tp+Vq, label ="T + V")
    plt.legend()
    plt.show()
    plt.plot(np.reshape(Tpa,len(Tpa)), label="Ta")
    plt.plot(np.reshape(Vqa,len(Vqa)),  label ="Va")
    plt.plot(Tpa+Vqa, label ="Ta + Va")
    plt.legend()
    plt.show()

def train_unknown(pq):
    
    K = 20
    h = 0.1
    sifts = 10000
    Ihat = 2000
    tau = 2/Ihat
    
    batches = import_batches()
    batch1 = batches[0]
    antB = 40
    
    
    bigbatch = {}
    bigbatch["Y"] = np.array([[],[],[]])
    bigbatch["c"] = np.array([])
    
    for i in range(antB):
        batch = batches[i]
        bigbatch["Y"] = np.append(bigbatch["Y"],batch["Y_"+pq],1)
        bigbatch["c"] = np.append(bigbatch["c"],batch["c_"+pq])
        
    Y = bigbatch["Y"]
    c = bigbatch["c"][:,np.newaxis]
    
    sc,inv = scale(c)
    
    sparameters = scale(c,returnParameters = True)
    
    
    inv_file = open(pq + "_unknown_inv.pkl", "wb")
    pickle.dump(sparameters, inv_file)
    inv_file.close()
    
    d_0 = Y.shape[0]
    d = d_0*2
    
    
    
    th = initialize_weights(d_0, d, K)
    
    JJ, th = stocgradient(sc, d, d_0, K, h, Y, th, tau, 1 , Ihat, sifts, True, pq + "_unknown_w.pkl")
    #JJ, th = stocgradient(sc, d, d_0, K, h, Y, th, tau, 1 , Ihat, sifts)
    
    plt.plot(JJ)
    plt.yscale("log")
    plt.show()
    
    
    th_file = open(pq + "_unknown_w.pkl", "wb")
    pickle.dump(th, th_file)
    th_file.close()
    
    

def test_unknown(pq):
    
    K = 20
    h = 0.1
    
    
    inv_file = open( pq + "_unknown_inv.pkl", "rb")
    inv = pickle.load(inv_file)
    inv_file.close()
    
    w_file = open(pq + "_unknown_w.pkl", "rb")
    th = pickle.load(w_file)
    w_file.close()
    
    batches = import_batches()
    batch1 = batches[0]
    antB = 49
    
    Y = batch1["Y_q"]
    d_0 = Y.shape[0]
    d = d_0*2
    
    for i in range(antB):
        plt.title("Batch: " + str(i) + ",   y = F(" + pq +")")
        testbatch = batches[i]
    
        tY = testbatch["Y_"+pq]
        
        z, yhat = F_tilde(tY, th, d_0, d, K, h)
        
        
        y = invscaleparameter(yhat, inv[0], inv[1], inv[2], inv[3])
        c = testbatch["c_"+pq]
        
        plt.plot(y,label ="y")
        plt.plot(c,label ="c")
        plt.legend()
        plt.show()
    
    


def model_unknown():   
    
    K = 20
    h = 0.1
    
    
    invp_file = open("p_unknown_inv.pkl", "rb")
    invp = pickle.load(invp_file)
    invp_file.close()
    
    invq_file = open("q_unknown_inv.pkl", "rb")
    invq = pickle.load(invq_file)
    invq_file.close()
    
    wp_file = open("p_unknown_w.pkl", "rb")
    thp = pickle.load(wp_file)
    wp_file.close()

    wq_file = open("q_unknown_w.pkl", "rb")
    thq = pickle.load(wq_file)
    wq_file.close()
    
    batches = import_batches()
    batch1 = batches[0]
    antB = 10
    
    Y = batch1["Y_q"]
    d_0 = Y.shape[0]
    d = d_0*2
    
    N = Y.shape[1]
    
    
    for i in range(antB):
        plt.title("Batch: " + str(i))
        testbatch = batches[i]
    
    
    
        pa = testbatch["Y_p"]
        Ta = testbatch["c_p"]
        qa = testbatch["Y_q"]
        Va = testbatch["c_q"]
        
        
        
        
        plt.plot(np.reshape(Ta,len(Ta)), label="Ta")
        plt.plot(np.reshape(Va,len(Va)),  label ="Va")
        plt.plot(Ta+Va, label ="Ta + Va")
        plt.legend()
        plt.show()
        
        
        z, yhatp = F_tilde(pa, thp, d_0, d, K, h)
        Tpa = invscaleparameter(yhatp, invp[0], invp[1], invp[2], invp[3])
        z, yhatq = F_tilde(qa, thq, d_0, d, K, h)
        Vqa = invscaleparameter(yhatq, invq[0], invq[1], invq[2], invq[3])
        
        plt.plot(np.reshape(Tpa,len(Tpa)), label="Tpa")
        plt.plot(np.reshape(Vqa,len(Vqa)),  label ="Vqa")
        plt.plot(Tpa+Vqa, label ="Tpa + Vqa")
        plt.legend()
        plt.show()
        
        
        """
        z, yhat = F_tilde(tY, th, d_0, d, K, h)
        
        
        y = invscaleparameter(yhat, inv[0], inv[1], inv[2], inv[3])
        ic = invt(tc)
        
        plt.plot(y,label ="y")
        plt.plot(ic,label ="c")
        plt.legend()
        plt.show()
        """

    
    

#train_two_body("p")
#train_two_body("q")
#test_two_body()

#train_nlp("p")
#train_nlp("q")
    
#test_nlp("p")
#test_nlp_grad("p")

#test_nlp("q")
#test_nlp_grad("q")

#model_nlp()
#model_two_body()
    
#train_unknown("p")
#train_unknown("q")

test_unknown("p")
test_unknown("q")   

#model_unknown()



    
    