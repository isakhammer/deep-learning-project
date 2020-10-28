#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:31:58 2020

@author: isakh
"""

def kepler(p,q):
    if (p.shape[0] != 2) or (q.shape[0] != 2):
        print("Wrong dimension")
        sys.exit(1)
    
    K = 0.5* (p[0]**2 + p[1]**2) 
    V = -1/np.sqrt(q[0]**2 + q[1]**2) 
    return K, V

def sqr_2(p,q):
    if (p.shape[0] != 2) or (q.shape[0] != 2):
        print("Wrong dimension")
        sys.exit(1)
    
    K = 0.5*p[0]**2 + 0.5*p[1]**2
    V = 0.5*p[0]**2 + 0.5*p[1]**2
    return K, V

def sqr_1(p,q):
    if (p.shape[0] != 1) or (q.shape[0] != 1):
        print("Wrong dimension")
        sys.exit(1)
    
    K = 0.5*p**2
    V = 0.5*q**2
    return K,V

def cos_1(p,q):

    if (p.shape[0] != 1) or (q.shape[0] != 1):
        print("Wrong dimension")
        sys.exit(1)
 
    K = 1 - np.cos(p)
    V = 1 - np.cos(q)
    return K,V