#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:59:16 2023

@author: robertgc
"""

from data import data
from lib import model
# from corner import corner
import numpy as np
import matplotlib.pyplot as plt


#%%
default = np.array([-6,1.04,.39])

#The data 
d = data()

#Model set up with data
m = model(X=d.X)

#%%

l = np.linspace(0,400)

inp = np.empty_like(default)
inp[:] = default[:]

# inp[1] *= 1.01

plt.plot(
    d.y,
    m.predict(inp),
    'o'
    )

plt.plot(l,l)