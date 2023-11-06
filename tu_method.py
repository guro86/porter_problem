#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:42:13 2023

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

#The data 
d = data()

#Model set up with data
m = model(X=d.X)
