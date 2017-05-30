# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# In[]
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns

# In[]
dim = 5
n_train = 1000
n_test = 400
var = 0.05
mean = 1.0/5.0
# In[]
def getDist(var, size):
    
    mean_n = [-mean]*dim
    mean_p = [mean]*dim
    ones = np.random.uniform(0.0,1.0,size) > 0.5
    y = np.zeros(size);
    y[ones] = 1
    y[True^ones] = -1
    X = np.zeros((size,dim))
    cov = np.identity(dim) * var
    D0 = np.random.multivariate_normal(mean_n, cov, size-np.sum(ones))
    D1 = np.random.multivariate_normal(mean_p, cov, np.sum(ones))
    j0=0
    j1=0
    for i in range(len(ones)):
        if ones[i]==False:
            X[i] = D0[j0]
            j0 = j0 + 1
        else:
            X[i] = D1[j1]
            j1 = j1 + 1
    return X,y
# In[]
def plotDist(X, a=1):
    sns.set(color_codes=True)
    sns.distplot(X[:,a]);

# In[]
X_train,y_train = getDist(var,n_train);
X_test,y_test   = getDist(var,n_test);
plotDist(X_train)
# In[]          
def loss(w,X,y):
    
def error(w,X,y):
    
def SGD(X,y):
    
    
    