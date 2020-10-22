#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:11:30 2020

@author: isakh
"""


import numpy as np
import os

def sigma(x, derivative=False):
   
    if (derivative):
        return 1 / np.cosh(x)**2 
    return np.tanh(x)

def eta(x, derivative=False):
    if (derivative):
        return (1/4)*(1 / np.cosh(x)**2 )
    return 0.5*(1 + np.tanh(x*0.5))


"""
Input:
    Z_k = (d, N)
    w = (d, 1)
    mu = (1, 1)

out: 
    \eta( Z_k^T w  + \mu ) = (I, 1) 
"""
def F_tilde(Y, th, d_0, d_k, K, h):
    
    Z = {}
    dsigma = {}
    
    I_d = np.identity(d_k)[:,:d_0]
    Z[0] = I_d@Y
    
    
    Z_hat= th["W"+str(0)]@Z[0]+th["b"+str(0)]
    
    Z[1] = Z[0] + h*sigma(Z_hat, False)
    
    dsigma[0] = sigma( Z_hat, True)
    
    for k in range(2, K+1):
        Z_hat = th["W"+str(k-1)]@Z[k-1]+th["b"+str(k-1)]
        Z[k] = Z[k-1] + h*sigma(Z_hat, False)
        dsigma[k-1] = sigma(Z_hat, True)
    
    #Z_hat = th["W"+str(K)]@Z[K]+th["b"+str(K)]
    #dsigma[K] = sigma(Z_hat, True)
    
    Upsilon = eta(Z[K].T@th["w"]+th["mu"])
    dUpsilon = eta(Z[K].T@th["w"]+th["mu"], derivative=True)
    
    # Maybe return dZ instead of deta and eta?
    return Z, Upsilon, dUpsilon, dsigma


def grad_J(c ,Y, th, d_0, d_k, K, h):
    
    I =  Y.shape[1]
    
    # Equation 5
    Z, Upsilon, dUpsilon, dsigma = F_tilde(Y, th, d_0, d_k, K, h)    
    
    # Equation 6
    J = 0.5*np.linalg.norm(Upsilon - c)**2

    # Initializing gradient
    dJ = {}

    # Equation (8)  
    dJ["mu"]   =  dUpsilon.T@(Upsilon - c)
        
    # Equation (9)
    dJ["w"] = Z[K]@((Upsilon - c)* dUpsilon)
        
    # Equation (10)
    P = np.zeros(( K+1, d_k, I))
    P[-1] = th["w"] @ ((Upsilon - c)* dUpsilon).T
        
    # Equation 11
    for k in range(K-1,-1,-1):
        P[k] = P[k+1] + h*th["W"+str(k)].T @ (dsigma[k] * P[k+1])  
    
    for k in range(0, K):
        # Equation 12
        dJ["W"+str(k)] = h*(P[k+1] * dsigma[k])@(Z[k]).T
        
        # Equation 13
        dJ["b"+str(k)] = h*(P[k+1] * dsigma[k])@np.ones((I,1))
    
    return J, dJ




    




"""
Imports data from a batch csv with format (t, q1, q2, q3, p1, p2, p3, K, V)

Output:
    n batches with following matrices:
        t: time (I, 1)
        Y_p: kinetic variables p1, p2, p3 (3, I)
        Y_q: potential variables q1, q2, q3 (3, I)
        c_p: total kinetic energy K (I, 1)
        c_q: total potential energy V (I, 1)

"""
def import_batches():
    n_batches = 49
    
    data_prefix = "datalist_batch_"
    data_path = os.path.join(os.path.dirname(__file__), "project_2_trajectories")
    
    batches = {}
    
    for i in range(n_batches):
        # assemble track import path
        batch_path = os.path.join(data_path, data_prefix + str(i) + ".csv")
        batch_data = np.loadtxt(batch_path, delimiter=',', skiprows=1)
        
        # np.newaxis is adding a dimension such that (I,) -> (I, 1)
        batch = {}
        batch["t"] = batch_data[:, 0, np.newaxis]
        batch["Y_q"] = batch_data[:, 1:4].T
        batch["Y_p"] = batch_data[:, 4:7].T
        batch["c_p"] = batch_data[:, 7, np.newaxis] 
        batch["c_q"] = batch_data[:, 8, np.newaxis] # potential energy
        
        batches[i] = batch

    return batches



def import_batches_example():
    
    
    batches = {}
    batch = {}   
    
    batch["Y_q"] = np.array([
                    [1,2,1],        	           
                    [1,1,2],        	       
                    [1,2,1],        	               
                    [1,2,1]        	          
                    ]).T	       
    batch["Y_p"] =  batch["Y_q"]  
    batch["c_q"] = np.array([[1,2,1,1]]).T	
   
    batch["c_p"] = batch["c_q"]
    batches[0] = batch
    
    return batches

    
def gradient_descent(batches, th, d_0, d_k, K, h, name, epochs):
    
    print("\n GRADIENT DESCENT - Variable: ", name)
    
    Y_key, c_key = None, None
    
    if name=="q":
        Y_key = "Y_q"
        c_key = "c_q"
    
    if name=="p":
        Y_key = "Y_p"
        c_key = "c_p"
        
    tau = 0.5
    tol = 0.01
    
    #for i in range(epochs):
    J = np.inf
    i = 0
    while i <= epochs :#and J < tol:
        i += 1
        for index in batches:
            Y = batches[index][Y_key]
            c = batches[index][c_key]
            
            J, dJ = grad_J(c, Y, th, d_0, d_k, K, h)
            
            for key in th:
                th[key] -= tau*dJ[key]
        print("Gradient descent - Variable: ", name, " Epoch : ", i, " Cost:", J )
    return th, J

def adams_method(batches, th, d_0, d_k, K, h, name, epochs):
    
    print("\n ADAMS METHOD - Variable: ", name)
   
    Y_key, c_key = None, None
    
    if name=="q":
        Y_key = "Y_q"
        c_key = "c_q"
    
    if name=="p":
        Y_key = "Y_p"
        c_key = "c_p"
        
    v = {} 
    m = {}
    
    beta_1, beta_2 =  0.9, 0.999
    alpha, epsilon = 0.01, 10**(-8)
    
    for key in th:
        v[key] = np.zeros(th[key].shape)   
        m[key] = np.zeros(th[key].shape)   
    
    for i in range(epochs):
    #while itr <= maxitr and err > tol:
            
        for index in batches:
            Y = batches[index][Y_key]
            c = batches[index][c_key]
            
            j = (i* len(batches) + index)
            
            J, dJ = grad_J(c, Y, th, d_0, d_k, K, h)
            
            for key in th:
                g = dJ[key]
                m[key] = beta_1*m[key] + (1- beta_1)*g
                v[key] = beta_2*v[key] + (1 - beta_2)*(g*g)
                mhat = m[key]/(1 - beta_1**j)
                vhat = v[key]/(1 - beta_2**j)
                th[key] -= alpha*mhat/(np.square(vhat) + epsilon)
                
            
        print("Adams Method - Variable: ", name, " Epoch : ", i, " Cost:", J )
    return th
    

def scale_batches(batches):
    def scale(Y, alpha= 0, beta=1):
        return  (1/(np.amax(Y)- np.amin(Y)))*(( np.amax(Y) -Y)*alpha + (Y - np.amax(Y))*beta)
    
    for index in batches:
        
        # scaling
        batches[index]["Y_q"] = scale(batches[index]["Y_q"])
        
        batches[index]["c_q"] = scale(batches[index]["c_q"])
        batches[index]["Y_p"] = scale(batches[index]["Y_p"])
        
        batches[index]["c_p"] = scale(batches[index]["c_p"])
    
    return batches


def train(batches, th_q, th_p, d_0, d_k, K, h, epochs, method):
    
    
    
    if method=="adam":
        th_p = adams_method(batches, th_p, d_0, d_k, K, h, name="p",  epochs=epochs)
        th_q = adams_method(batches, th_q, d_0, d_k, K, h, name="q", epochs=epochs)
    
    elif method=="gradient_descent":
        th_p = gradient_descent(batches, th_p, d_0, d_k, K, h, name="p", epochs=epochs)
        th_q = gradient_descent(batches, th_q, d_0, d_k, K, h, name="q", epochs=epochs)
    
    else:
        print("No training method specified.")
        
    return th_q, th_p
        

def initialize_weights(d_0, d_k, K):
    th = {}
    
    if (K>1):
        th["W"+str(0)] = 2*np.random.random((d_k, d_k )) - 1
        th["b"+str(0)] = np.random.random((d_k, 1))
        
        for i in range(1, K):
            th["W"+str(i)] = 2*np.random.random(( d_k, d_k)) - 1
            th["b"+str(i)] = np.random.random((d_k, 1))
            
        th["w"] = 2*np.random.random((d_k, 1 )) - 1
        th["mu"] = np.random.random((1, 1))
    else:
        th["w"] = 2*np.random.random((d_k, 1 )) - 1
        th["mu"] = np.random.random((1, 1))
    
    return th

    

def main():
    
    K = 10
    h = 1/2
    d_0 = 3
    d_k = 10    
    epochs = 10000
    
    batches = import_batches_example() 
    #batches =  import_batches()
    
    #batches = scale_batches( batches)
    
    th_p = initialize_weights(d_0, d_k, K)
    th_q = initialize_weights(d_0, d_k, K)
    
    train( batches, th_q, th_p, d_0, d_k, K, h, epochs, method="gradient_descent")
    #train( batches, th_q, th_p, d_0, d_k, K, h, epochs, method="adam")    


def test():
    batches = import_batches()
    
    
    
    K = 10
    h = 1/2
    d_0 = 3
    d_k = 10    
    
    Y_key = "Y_p"
    
    th = initialize_weights(d_0, d_k, K)
    for index in batches:
        Y = batches[index][Y_key]
        f =  F_tilde(Y, th, d_0, d_k, K, h)
    
    
    


main()
#test()
    