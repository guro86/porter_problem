#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:05:55 2023

@author: robertgc
"""

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

rng = np.random.default_rng()

d = data()

n = 10000

mu = np.array(
    [3.333e-3,1.015,0.383]
    )

sigma = np.array(
    [5.66e-4,4.27e-3,4.24e-3]
    )


columns = [fr'$\theta_{i+1}$' for i in range(3)]

samp = pd.DataFrame(
    rng.normal(mu,sigma,(n,3)),
    columns = columns)


corner(samp)

m = model(
    X = d.X
    )

preds = m.predict(
    samp
    )

plt.savefig('tu-posterior.pdf')



#%%

l = np.linspace(0,400,2)
plt.errorbar(
    d.y,
    preds.mean(axis=1),
    yerr = preds.std(axis=1),
    fmt='o'
    )

plt.plot(l,l)

plt.xlabel('Measuremd Nu')
plt.ylabel('Predicted Nu')

plt.savefig('tu-methodpropagation.pdf')

#%%


mean = preds.mean(axis=1)
std = preds.std(axis=1)

sns.histplot(
    (mean - d.y)/std,
    kde=True
    )

chi2 = np.sum((mean - d.y)**2/std**2) / len(d.y)

print(f'chi2: {chi2:.2f}')

plt.savefig('tu-resid.pdf')