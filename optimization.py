#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:18:48 2023

@author: robertgc
"""

from scipy.stats import norm
from data import data
from lib import model
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

#%%

#Load data and set up model
d = data()
m = model(X=d.X)

#Likelihood with observations and meas_unc
like = norm(
    loc = d.y,
    scale = 0.03*d.y
    )

#Logprob fn
log_prob_fn = lambda theta: like.logpdf(
        m.predict(theta)
    ).sum()


theta0 = np.ones(3)

sol = minimize(
    lambda theta: -log_prob_fn(theta),
    theta0,
    method='Nelder-Mead'
    )

print(sol.message)

l = np.linspace(0,400)

plt.plot(
    d.y,
    m.predict(sol.x),
    'o'
    )

plt.plot(l,l)


#X and y labels
plt.xlabel('Nu measured')
plt.ylabel('Nu predicted')

plt.title('Optimization')

plt.savefig('optimization.pdf')

plt.show()