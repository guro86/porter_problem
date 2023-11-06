#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:36:58 2023

@author: robertgc
"""
import emcee
from lib.stats import norm
from data import data
from lib import model
from corner import corner
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%

nwalkers = 6
ndim = 3

nsteps = 5000

#Initial state
initial_state = np.random.randn(nwalkers,ndim) * 1e-6 \
    + np.log(np.array([2.5e-3,1,.4]))

#The data 
d = data()

#Model set up with data
m = model(X=d.X)

#Likelihood with observations and meas_unc
like = norm(
    loc = d.y,
    scale = 0.01*d.y
    )

#Function that is proportional to logp 
log_prob_fn = lambda theta: like.logpdf(
        m.predict_exp(theta)
    ).sum()

#Stretch move just to avoid tuning 
moves = emcee.moves.StretchMove()

moves = emcee.moves.GaussianMove(cov=0.001**2)

#A sampler 
sampler = emcee.EnsembleSampler(
    nwalkers, 
    ndim, 
    log_prob_fn, 
    moves=moves
    )

#Sample and save the end state
state = sampler.run_mcmc(
    initial_state,
    nsteps, 
    progress = True,
    skip_initial_state_check=True
    )


#%%

#Column names 
columns = [fr'$\theta_{i}$' for i in range(3)]

#Get chain data in data frame 
chain = pd.DataFrame(
    sampler.get_chain(flat=True,discard=500),
    columns = columns
    )

#Corner plot
corner(chain)

#Show plot
plt.show()

#%%

#Print covariance matirx of the posterior
print(
      chain.cov()
      )

#As well as mean
print(
      chain.std().values
      )

#%%

#equality line 
l = np.linspace(0,400)

#Propagate the chain
preds = m.predict_exp(chain.values)

preds_noise = preds + np.random.randn(*preds.shape)*d.y[:,None]*.01

#Calculate mean and std. deviations
#Probably very small
pred = preds.mean(axis=1)
pred_std = preds.std(axis=1)

pred_noise_std = preds_noise.std(axis=1)


#Error bar
plt.errorbar(
    d.y,
    pred ,
    yerr = 2 * pred_noise_std,
    ls = "None",
    capsize=2,
    )


#Error bar
plt.errorbar(
    d.y,
    pred ,
    yerr = 2 * pred_std,
    ls = "None",
    capsize=5,
    )

#X and y labels
plt.xlabel('Nu measured')
plt.ylabel('Nu predicted')

#Plot equality line
plt.plot(l,l)

#Show the plot
plt.show()