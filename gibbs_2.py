#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 08:42:29 2023

@author: robertgc
"""

# from scipy.stats import norm, uniform
from lib.stats import norm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from corner import corner
from lib.transforms import logit
from data import data
from lib import model
import seaborn as sns


#%%

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

meas_v = data_object.y

#Number of samples to generate
nsamp = 2000

#Burnin samples
nburn = 1000

#Dimensions and number of experiments
nexp = 31

#Dimensions
dims_active = np.array([0,1,2])

ndims = 3

ndims_active = 3

#Mean and sigmas


default = np.array([-6.02110051,  0.04223386, -0.93515385])
#Local parameterss
X = (np.ones((nexp,ndims))*default)

R = lambda m: 0.02

#Likelihoods
likes = [norm(loc=m,scale=R(m)) for m in meas_v]

#Proposal 
# propose = norm(scale=0.1)

# propose_mu = norm(scale=0.1)
# propose_sig = norm(scale=0.1)

scale_sig = np.array([.1,0.001,0.0001])
scale_mu = np.array([.1,0.001,0.001])

scale_X = np.array([.1,0.001,0.001])

m = model(X=data_object.X)

rng = np.random.default_rng()


mu = default
sig = scale_sig



#%%

def update_dim(X,d,e,mu,sig):
    
    hier = norm(loc=mu,scale=sig)
    
    pred = m.predict_exp(X[None,:],e).flatten()
    
    logp = likes[e].logpdf(pred) + \
        hier.logpdf(X[dims_active]).sum()
    
    X_cand = np.empty(len(X))
    X_cand[:] = X[:]
    # X_cand[d] += propose.rvs()
    X_cand[d] += rng.normal(0,scale_X[d])
    
    pred_cand = m.predict_exp(
        X_cand[None,:],e
        ).flatten()
    
    logp_cand = likes[e].logpdf(pred_cand) + \
        hier.logpdf(X_cand[dims_active]).sum()
    
    u = rng.random(1)
    
    if logp_cand - logp > np.log(u):

        X[:] = X_cand[:]
        
    return X

def update_hyper(X,mu,sig,d):
    
    mu_cand = mu + rng.normal(0,scale_mu[d])
    
    logp = norm(loc=mu,scale=sig).logpdf(X).sum()
    
    logp_cand = norm(loc=mu_cand,scale=sig).logpdf(X).sum()
    
    u = rng.random(1)
    
    if logp_cand - logp > np.log(u):
        mu = mu_cand
        logp = logp_cand
        
    sig_cand = sig + rng.normal(0,scale_sig[d])
    
    logp_cand = norm(loc=mu,scale=sig_cand).logpdf(X).sum()
    
    u = rng.random(1)
    
    if logp_cand - logp > np.log(u):
        sig = sig_cand
        logp = logp_cand
        
    
    return mu, sig
    
#%%


Xs = np.empty((nsamp,nexp,ndims))
sigs = np.empty((nsamp,ndims_active))
mus = np.empty((nsamp,ndims_active))

#Loop over samples
for s in tqdm(range(nsamp)):

    #Loop dimensions
    for di,d in enumerate(dims_active):
    
        #Loop over experiments
        for e in range(nexp):
                    
            #Update the current dimension
            X[e,:] = update_dim(X=X[e],d=d,e=e,mu=mu,sig=sig)
            
        #Update hyper-parameters
        mu[di], sig[di] = update_hyper(
            X=X[:,d],mu=mu[di],sig=sig[di],d=d
            )
        
    Xs[s] = X
    mus[s] = mu
    sigs[s] = sig

#%%

nburn = 500

columns = [fr'$\theta_{i}$' for i in range(3)]

Xp = pd.DataFrame(
    rng.normal(mus[nburn:],sigs[nburn:]),
    columns = columns
    )
corner(np.exp(Xp))

plt.savefig('gibbs-marginalized.pdf')

#%%

columns = [fr'$\mu_{i}$' for i in range(3)]
corner(
       pd.DataFrame(
           mus[nburn:],
           columns=columns
           )
       )

plt.savefig('gibbs-mu.pdf')

#%%

columns = [fr'$\sigma_{i}$' for i in range(3)]
corner(
       pd.DataFrame(
           sigs[nburn:],
           columns=columns
           )
       )

plt.savefig('gibbs-sigma.pdf')

#%%
# corner(np.exp(np.concatenate(Xs[500:])))

#%%

l = np.linspace(0,400,2)

preds = m.predict_exp(Xp.values)
mean = preds.mean(axis=1)
std = preds.std(axis=1)

plt.errorbar(
    meas_v,
    mean,
    yerr = std,
    fmt='o',
    label = 'Mean with 1-sig parameter unc.'
    )

plt.legend()


plt.xlabel('Measuremd Nu')
plt.ylabel('Predicted Nu')

plt.plot(l,l)

plt.savefig('gibbs.pdf')

#%%

sns.histplot(
    (mean - meas_v) / std,
    kde = True
    )

plt.savefig('gibbs-residuals.pdf')


#%%

chi2 = np.sum((mean - meas_v)**2/std**2) / len(meas_v)

print(f'chi2: {chi2:.2f}')