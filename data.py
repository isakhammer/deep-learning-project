#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:09:31 2020

@author: isakh
"""
import numpy as np
import os

"""
def generate_synthetic_batches(I,func = "2sqr"):
    
    batches = {} 
    batches["V"], batches["K"] = {}, {}
    
    batch = {} 
    
    if func == "2sqr":
        
        d_0 = 2
        Y1 = np.linspace(-1,1,I)
        Y2 = np.linspace(1,-1,I)
    
        batch["Y"] = np.array([Y1,Y2])
        batch["Y"] = np.array([[1,1],
                              [1,2]])
        
        batch["Y"] = np.random.uniform(high=2, low=-2, size=(d_0,I) )    
        batch["c"] = 0.5*batch["Y"][0,:]**2 + 0.5*batch["Y"][1,:]**2
        batch["c"] = batch["c"][:, np.newaxis]
        
    
    elif func == "1sqr":
        d_0 = 1
        batch["Y"] = np.random.uniform(high=2, low=-2, size=(d_0,I) )
        batch["c"] = 0.5*(batch["Y"])**2
        batch["c"] = batch["c"].T
        
    elif func == "1cos":
        d_0 = 1
        batch["Y"] = np.random.uniform(high=np.pi/3, low=-np.pi/3, size=(d_0,I) )
        batch["c"] = 1 - np.cos(batch["Y"])
        batch["c"] = batch["c"].T
    
    else:
        raise Exception("Not axeped func")
    
    batches["K"][0] = batch
    batches["V"][0] = batch
    return batches
"""      
        
def generate_batches(I, n_batches, d_0, func, high=2, low=-2):
    batches = {} 
    batches["V"], batches["K"] = {}, {}
    for i in range(n_batches):
        V_batch, K_batch = {}, {}
        V_batch["Y"] = np.random.uniform(high, low, size=(d_0, I) )    
        K_batch["Y"] = np.random.uniform(high, low, size=(d_0, I) )    
        K_batch["c"], V_batch["c"] = func( K_batch["Y"], V_batch["Y"])
        
        V_batch["c"] = V_batch["c"][:, np.newaxis]
        K_batch["c"] = K_batch["c"][:, np.newaxis]
       
        batches["V"][i] = V_batch
        batches["K"][i] = K_batch
    
    return batches

    

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
        #batch["t"] = batch_data[:, 0, np.newaxis]
        batch["Y_q"] = batch_data[:, 1:4].T
        #batch["Y_p"] = batch_data[:, 4:7].T
        #batch["c_p"] = batch_data[:, 7, np.newaxis] 
        batch["c_q"] = batch_data[:, 8, np.newaxis] # potential energy
        batches[i] = batch

    return batches

def import_one_batch():
   
    data_prefix = "datalist_batch_"
    data_path = os.path.join(os.path.dirname(__file__), "project_2_trajectories")
    
    batches = {}
    
    i = 0
    # assemble track import path
    batch_path = os.path.join(data_path, data_prefix + str(i) + ".csv")
    batch_data = np.loadtxt(batch_path, delimiter=',', skiprows=1)
        
    # np.newaxis is adding a dimension such that (I,) -> (I, 1)
    batch = {}
    #batch["t"] = batch_data[:, 0, np.newaxis]
    batch["Y_q"] = batch_data[:, 1:4].T
    #batch["Y_p"] = batch_data[:, 4:7].T
    #batch["c_p"] = batch_data[:, 7, np.newaxis] 
    batch["c_q"] = batch_data[:, 8, np.newaxis] # potential energy
        
    batches[0] = batch
    return batch
    
        
        
        
        
        
