#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:31:18 2020

@author: isakh
"""


import numpy as np
import os


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
    batch["t"] = batch_data[:, 0, np.newaxis]
    batch["Y_q"] = batch_data[:, 1:4].T
    batch["Y_p"] = batch_data[:, 4:7].T
    batch["c_p"] = batch_data[:, 7, np.newaxis] 
    batch["c_q"] = batch_data[:, 8, np.newaxis] # potential energy
        
    batches[0] = batch

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




def generate_synthetic_batches(model="nonlinear_pendulum"):
    
    batches = {}
    batch = {}   
    I = 4000
    
    if model=="nonlinear_pendulum":
        d = 1
        batch["Y_q"] = np.random.random((d, I))     
        batch["Y_p"] = np.random.random((d, I)) 
        batch["c_q"] = 0.5*batch["Y_p"]**2
        batch["c_p"] = (1 - np.cos(batch["Y_q"]))
        batches[0] = batch
        
    return batches
    