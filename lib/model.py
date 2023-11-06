#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:14:53 2023

@author: robertgc
"""

import numpy as np

class model():
    
    def __init__(self,X):
        
        self.X = X
        
    def predict_exp(self,theta,e=None):
        return self.predict(np.exp(theta),e)
    
    def predict(self,theta,e=None):
        
        X = self.X
        
        if e is not None: 
            X = X[e]
        
        exponents = theta.T[-2:]
        factor = theta.T[0].T
        
        pred = factor * np.exp(np.log(X) @ exponents)
        
        return np.squeeze(pred)

        
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    import sys
    sys.path.append('../')

    #Import data    
    from data import data
    
    #Load data
    d = data()
    
    m = model(X=d.X)
    
    theta = np.array([
        [1.e-3,1.,.4],
        [1.e-3,1.,.4],
        ])
    
    print(
        m.predict(theta,2)
        )