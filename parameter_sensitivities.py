#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:32:12 2020

@author: isakh
"""

from nn_2 import *
from data import *    
import matplotlib.pyplot as plt
import os

def tau_sensitivity(method="gd"):
           
    I = 100     
    b = generate_synthetic_batches(I)
    
    c, inv = scale(b["c"])
    Y = b["Y"]
    #Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = [ 2, 1, 0.5, 0.25, 0.1, 0.01]
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        tau = var[i]
        th = initialize_weights(d_0, d, K)
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method=method)
        plt.plot(it, JJ, label="tau: "+ str(tau))
    
    plt.title("Tau Sensitivity Analysis for " + method)
    plt.legend()
    plt.show()
    
def tauI_sensitivity(I, method="gd"):            
    
    b = generate_synthetic_batches(I)
    
    c, inv = scale(b["c"])
    Y = b["Y"]
    #Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = np.array([ 40, 20, 10, 5, 2, 0.2])/I
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        tau = var[i]
        th = initialize_weights(d_0, d, K)
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method=method)
        plt.plot(it, JJ, label="tau: "+ str(tau*I) + "/I")
    
    plt.title("Tau Sensitivity Analysis for " + method + ", I: "+str(I))
    plt.legend()
    plt.show()

def alpha_sensitivity(method="adam"):
                
    I = 100
    b = generate_synthetic_batches(I)
    
    c, inv = scale(b["c"])
    Y = b["Y"]
    #Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = [0.75*10**-4, 0.5*10**-4, 0.35*10**-4, 0.75*10**-5, 0.5*10**-5, 0.25*10**-6]
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        alpha = var[i]
        th = initialize_weights(d_0, d, K)
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method=method, alpha=alpha)
        plt.plot(it, JJ, label="alpha: "+ str(alpha))
    
    plt.title("Alpha Sensitivity Analysis for " + method)
    plt.legend()
    plt.show()
    
def alphaI_sensitivity(I, method="adam"):            
    
    b = generate_synthetic_batches(I)
    
    c, inv = scale(b["c"])
    Y = b["Y"]
    d_0 = Y.shape[0]
    
    var = np.array( [0.75*10**-4, 0.5*10**-4, 0.35*10**-4, 0.75*10**-5, 0.5*10**-5, 0.25*10**-6])/I
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        alpha = var[i]
        th = initialize_weights(d_0, d, K)
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method=method, alpha=alpha)
        plt.plot(it, JJ, label="alpha: "+ str(alpha*I) + "/I")
    
    plt.title("Alpha Sensitivity Analysis for " + method + ", I: "+str(I))
    plt.legend()
    plt.show()

def I_selection(var="tau", method="gd"):       
    I = [5, 10, 15, 20, 40, 80, 160, 320]
    for i in I:
        if var == "tau":
            tauI_sensitivity(i)
        elif var == "alpha":
            alphaI_sensitivity(i)
        else:
            print("no var")
    

def K_sensitivity(method="gd"):
             
    b = generate_synthetic_batches(I)
    
    c, inv = scale(b["c"])
    Y = b["Y"]
    #Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = var = [ 4, 6, 10, 14, 17, 20, 30]
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        K = var[i]
        th = initialize_weights(d_0, d, K)
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method=method)
        plt.plot(it, JJ, label="K: "+ str(var[i]))
    
    plt.title("K Sensitivity Analysis for " + method)
    plt.legend()
    plt.show()

    

def h_sensitivity(method="gd"):
             
    b = generate_synthetic_batches(I)
    
    c, inv = scale(b["c"])
    Y = b["Y"]
    #Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = var = [ 0.14, 0.12, 0.1, 0.07, 0.05, 0.01]
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        h = var[i]
        th = initialize_weights(d_0, d, K)
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method=method)
        plt.plot(it, JJ, label="h: "+ str(var[i]))
    
    plt.title("h Sensitivity Analysis for " + method)
    plt.legend()
    plt.show()


def d_sensitivity(method="gd"):
             
    b = generate_synthetic_batches(I)
    
    c, inv = scale(b["c"])
    Y = b["Y"]
    #Y = scale(b["Y"])
    d_0 = Y.shape[0]
    
    var = var = [ 2, 3, 4, 5,6,7,8, 10 ]
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        d = var[i]
        th = initialize_weights(d_0, d, K)
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method=method)
        plt.plot(it, JJ, label="d: "+ str(var[i]))
    
    plt.title("d Sensitivity Analysis for " + method)
    plt.legend()
    plt.show()

def I_sensitivity(method="gd"):
    max_it = 3000             
    var = var = [5, 10, 15, 20, 40, 80, 160, 320]
    it = np.arange(0,max_it+1)
    
    for i in range(len(var)):    
        I = var[i]
        b = generate_synthetic_batches(I)
        c, inv = scale(b["c"])
        Y = b["Y"]
        #Y = scale(b["Y"])
        d_0 = Y.shape[0]
    
        print("I:", I)
        th = initialize_weights(d_0, d, K)
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, print_it=False, method=method)
        JJ = JJ/JJ[0]
        plt.plot(it, JJ, label="I: "+ str(var[i]))
    
    #plt.yscale("log")
    plt.title("I Sensitivity Analysis for " + method)
    plt.yscale("log")
    plt.legend()
    plt.show()
    
    


# Default values based on the analysis
K = 20
h = 0.1
d_0 = 2
d = 4
I = 600
max_it = 1000
tau = 0.1

"""

K_sensitivity(method="gd")
h_sensitivity(method="gd")
d_sensitivity(method="gd")
I_sensitivity(method="gd")
I_selection(method="gd")

K_sensitivity(method="adam")
h_sensitivity(method="adam")
d_sensitivity(method="adam")
I_sensitivity(method="adam")
I_selection(method="adam")
"""
#tau_sensitivity(method="gd")
#alpha_sensitivity(method="adam")

I_selection(var="tau", method="gd")      
I_selection(var="alpha", method="gd")