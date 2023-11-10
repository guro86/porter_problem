#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:19:38 2023

@author: robertgc
"""

from data import data
from lib import model
from corner import corner
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from lib.stats import norm

#%%

#Steps to be taken
nsteps = 5000

#Default values
default = np.log(
    np.array([0.00397991, 0.98612586, 0.41062651])
    )

#Data and the model 
data_object = data()

#Set up a model with X data
model_object = model(
    X=data_object.X
    )

#Likelihoods with observations and meas_unc
likes = [
    norm(loc = loc, scale = 1e-6) for loc in data_object.y
    ]

#%%

#Samples to take
nsamp = 2000
nexp = len(data_object.y)
ndim = 3

#Hyper parameters
loc = default

#Random generator 
rng = np.random.mtrand.RandomState()

#Xvalues
X = rng.randn(nexp,ndim)*1e-10 + default

#Proposal_scales, subject to tuning later

scale = np.array([0.015,0.0012,0.0025])

#Candidate vector, initialize before loop for speed
X_cand = np.ones(3)

Xs = np.empty((nsamp,nexp,ndim))
locs = np.empty((nsamp,ndim))
scales = np.empty((nsamp,ndim))

locs[:] = loc
scales[:] = np.array([.1,0.001,0.0001])



scale_X = np.array([.1,0.001,0.0001])
scale_scale = np.array([.1,0.001,0.0001])
scale_loc = np.array([.1,0.001,0.0001])


#The outer loop
for s in tqdm(np.arange(nsamp)):
  
    #Loop the dimensions
    for d in np.arange(ndim):  
  
        #Loop experiments
        for e in np.arange(nexp):
                
            #Candiate of X
            x_cand = X[e,d] + rng.randn() * scale_X[d]
            
            #Candidate vector
            X_cand[:] = X[e,:]
            X_cand[d] = x_cand
            
            #Predict experiment with current vector
            pred = model_object.predict_exp(X[e,:],e)
            
            #Predict experiment with candidate vector
            pred_cand = model_object.predict_exp(X_cand,e)
            
            #Log pred
            log_pred = likes[e].logpdf(
                pred
                ) + norm(loc=loc[d],scale=scale[d]).logpdf(X[e,d])
            
            #Log candidate pred
            log_pred_cand = likes[e].logpdf(
                pred_cand
                ) + norm(loc=loc[d],scale=scale[d]).logpdf(x_cand)
            
            #Logdiff
            logdiff = log_pred_cand - log_pred
            
            #Log u
            logu = np.log(rng.rand())
            
            #If logdiff greater, accept
            if logu < logdiff:
                X[e,:] = X_cand
                
                
        scale_cand = scale[d] + rng.rand() * scale_scale[d]
        
        log_sig = norm(
            loc=loc[d],scale=scale[d]).logpdf(X[:,d]).sum()
                
        log_sig_cand = norm(
            loc=loc[d],scale=scale_cand).logpdf(X[:,d]).sum()
        
        #Log u
        logu = np.log(rng.rand())
        
        #Logdiff
        logdiff = log_sig_cand - log_sig
        
        #If logdiff greater, accept
        if logu < logdiff:
            scale[d] = scale_cand
        
        loc_cand = loc[d] + rng.rand() * scale_loc[d]
        
        log_loc = norm(
            loc=loc[d],scale=scale[d]).logpdf(X[:,d]).sum()
                
        log_loc_cand = norm(
            loc=loc_cand,scale=scale[d]).logpdf(X[:,d]).sum()
        
        #Log u
        logu = np.log(rng.rand())
        
        #Logdiff
        logdiff = log_loc_cand - log_loc
        
        #If logdiff greater, accept
        if logu < logdiff:
            loc[d] = loc_cand
            
    scales[s,:] = scale
    locs[s,:] = loc
    
    Xs[s] = X
    
#%%

l = np.linspace(0,400)

pred = model_object.predict_exp(np.concatenate(Xs))

mean_pred = pred.mean(axis=1)
# mean_pred = model_object.predict(default)
std_pred = pred.std(axis=1)

plt.errorbar(
    data_object.y,
    mean_pred,
    yerr = std_pred,
    fmt= 'o'
    )

plt.plot(l,l)

#%%
l = np.linspace(0,400)

pred = model_object.predict_exp(np.concatenate(Xs))

mean_pred = pred.mean(axis=1)
std_pred = pred.std(axis=1)

plt.plot(
    data_object.y,
    mean_pred+2*std_pred,
    'o'
    )
plt.plot(l,l)