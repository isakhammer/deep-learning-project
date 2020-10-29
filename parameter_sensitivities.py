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

def tau_sensitivity():
                
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
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method="gd")
        plt.plot(it, JJ, label="tau: "+ str(tau))
    
    plt.savefig(file_paths["tau_sensitivity"] )
    plt.title("Tau Sensitivity Analysis")
    plt.legend()
    plt.show()
    
def tauI_sensitivity(I):
                
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
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method="gd")
        plt.plot(it, JJ, label="tau: "+ str(tau*I) + "/I")
    
    plt.savefig(file_paths["tau_sensitivity"] )
    plt.title("Tau Sensitivity Analysis, I: "+str(I))
    plt.legend()
    plt.show()
    
def I_selection():       
    var = [5, 10, 15, 20, 40, 80, 160, 320]
    for I in var:
        tauI_sensitivity(I)
    

def K_sensitivity():
             
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
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method="gd")
        plt.plot(it, JJ, label="K: "+ str(var[i]))
    
    plt.title("K Sensitivity Analysis")
    plt.legend()
    plt.show()
    plt.savefig(file_paths["K_sensitivity"] )

    

def h_sensitivity():
             
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
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method="gd")
        plt.plot(it, JJ, label="h: "+ str(var[i]))
    
    plt.title("h Sensitivity Analysis")
    plt.legend()
    plt.show()
    plt.savefig(file_paths["h_sensitivity"] )


def d_sensitivity():
             
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
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, method="gd")
        plt.plot(it, JJ, label="d: "+ str(var[i]))
    
    plt.title("d Sensitivity Analysis")
    plt.legend()
    plt.show()
    plt.savefig(file_paths["d_sensitivity"] )

def I_sensitivity():
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
        JJ,th = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, print_it=False, method="gd")
        JJ = JJ/JJ[0]
        plt.plot(it, JJ, label="I: "+ str(var[i]))
    
    #plt.yscale("log")
    plt.title("I Sensitivity Analysis")
    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.savefig(file_paths["I_sensitivity"] )
    
    

def Ib_sensitivity():
    max_it = 3000            
    var = [ 5, 10, 15, 20, 40, 80, 160, 320]
    it = np.arange(0,max_it+1)
    bsize = 10
    
    for i in range(len(var)):    
        I = var[i]
        b = generate_synthetic_batches(I)
        c, sa, sb, salpha, sbeta = scale(b["c"])
        Y = b["Y"]
        #Y = scale(b["Y"])
        d_0 = Y.shape[0]
    
        print("I:", I)
        th = initialize_weights(d_0, d, K)
        
        if bsize <= I:
            numB = int(I/bsize)
            JJ = np.array([])
            for i in range(numB):
                dJJ, th = train(c[i*bsize:(i+1)*bsize], d, d_0, K, h, Y[:,i*bsize:(i+1)*bsize], th, tau, int(max_it/numB), print_it=False)
                JJ = np.append(JJ,dJJ)
                """
                if i != 0:
                    it = np.append(it,it[-1]+1)
                """
                
        else:
            JJ = train(c, d, d_0, K, h, Y, th, tau=tau, max_it=max_it, print_it=False) 
        
        #plt.plot(it, JJ, label="I: "+ str(I))
        plt.plot(JJ, label="I: "+ str(I))
    
    #plt.yscale("log")
    plt.title("I Sensitivity Analysis")
    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.savefig(file_paths["I_sensitivity"] )



file_paths = {}
file_paths["output"] = os.path.join(os.path.dirname(__file__), "output" ) 
os.makedirs(file_paths["output"], exist_ok=True) 
file_paths["K_sensitivity"] = os.path.join(file_paths["output"], "K_sensitivity.png")
file_paths["tau_sensitivity"] = os.path.join(file_paths["output"], "tau_sensitivity.png")
file_paths["h_sensitivity"] = os.path.join(file_paths["output"], "h_sensitivity.png")
file_paths["d_sensitivity"] = os.path.join(file_paths["output"], "d_sensitivity.png")
file_paths["I_sensitivity"] = os.path.join(file_paths["output"], "I_sensitivity.png")


# Default values based on the analysis
K = 20
h = 0.1
d_0 = 2
d = 4
I = 600
max_it = 300
tau = 0.1


#K_sensitivity()
tau_sensitivity()
#h_sensitivity()
#d_sensitivity()
#I_sensitivity()
I_selection()
#Ib_sensitivity()