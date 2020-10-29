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
    
    
    
    
    
    
    
    

test_two_body()
#train_two_body()
    
    