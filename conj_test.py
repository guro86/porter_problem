#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:16:58 2023

@author: robertgc
"""

import numpy as np
from scipy.stats import norm
import emcee 
import matplotlib.pyplot as plt


scale = .1


np.random.seed(1)
n = 100
x = np.random.randn(n) * scale


nwalkers = 2
ndim = 1


nsteps = 2000

initial_state = norm().rvs((nwalkers,ndim))

log_prob_fn = lambda loc: norm.logpdf(x,loc=loc,scale=scale).sum()

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn)

state = \
    sampler.run_mcmc(
        initial_state=initial_state,
        nsteps=nsteps,
        progress=True
        )
    
    
chain = sampler.get_chain(discard=500,flat=True)

plt.hist(chain)