#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:09:31 2020

@author: isakh
"""
import numpy as np

def generate_synthetic_batches(I,func = "2sqr"):
    
    batch = {} 
    
    if func == "2sqr":
        
        d_0 = 2
        """
        Y1 = np.linspace(-1,1,I)
        Y2 = np.linspace(1,-1,I)
    
        batch["Y"] = np.array([Y1,Y2])
        batch["Y"] = np.array([[1,1],
                              [1,2]])
        """
        
        batch["Y"] = np.random.uniform(high=2, low=-2, size=(d_0,I) )    
    
    
        batch["c"] = 0.5*batch["Y"][0,:]**2 + 0.5*batch["Y"][1,:]**2
        batch["c"] = batch["c"][:, np.newaxis]
        
        return batch
    
    elif func == "1sqr":
        d_0 = 1
        
        batch["Y"] = np.random.uniform(high=2, low=-2, size=(d_0,I) )
        batch["c"] = 0.5*(batch["Y"])**2
        batch["c"] = batch["c"].T
        
        return batch
    
    elif func == "1cos":
        d_0 = 1
        
        batch["Y"] = np.random.uniform(high=np.pi/3, low=-np.pi/3, size=(d_0,I) )
        batch["c"] = 1 - np.cos(batch["Y"])
        batch["c"] = batch["c"].T
        
        return batch
    
    else:
        raise Exception("Not axeped func")
    
        
        
        
        
        
