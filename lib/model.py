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
        
        #Predict using exp input
        return self.predict(np.exp(theta),e)
    
    def predict(self,theta,e=None):
        
        #Get the intputs
        X = self.X
        
        #If just the e-the experiement is requested
        if e is not None: 
            X = X[e]
        
        #get the expponents 
        exponents = theta.T[-2:]
        
        #Get the factor
        factor = theta.T[0].T
        
        #Predict
        pred = factor * np.exp(np.log(X) @ exponents)
        
        #Return squeezeed
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