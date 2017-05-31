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
dim_X = 5
dim_C = 6
T = 1000
n_train = T-1
n_test = 400
var = 0.05
mean = 1.0/5.0
dist_type = "box"
sample_no = 0
# In[]
def getDist(var, size):
    
    mean_n = [-mean]*dim_X
    mean_p = [mean]*dim_X
    ones = np.random.uniform(0.0,1.0,size) > 0.5
    y = np.zeros(size);
    y[ones] = 1
    y[True^ones] = -1
    X = np.zeros((size,dim_X))
    cov = np.identity(dim_X) * var
    #D0 = np.random.multivariate_normal(mean_n, cov, size-np.sum(ones))
    #D1 = np.random.multivariate_normal(mean_p, cov, np.sum(ones))
    D0 = np.random.multivariate_normal(mean_n, cov, size/2)
    D1 = np.random.multivariate_normal(mean_p, cov, size/2)
    j0=0
    j1=0
    '''
    for i in range(len(ones)):
        if ones[i]==False:
            X[i] = D0[j0]
            j0 = j0 + 1
        else:
            X[i] = D1[j1]
            j1 = j1 + 1
    '''
    for i in range(size):
        if i%2==0:
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
dist_X,dist_y = getDist(var,n_train+n_test);
#X_test,y_test   = getDist(var,n_test);
plotDist(X_train)
# In[]         
def isOutOfSet_hypercube(x):
    for x_i in x:
        if(abs(x_i) >  1):
            return True
    return False
    
def isOutOfSet_ball(x):
    if(np.linalg.norm(x) >  1):
            return True
    return False
# In[]
def projectBack_hypercube(x):
    
    for i in range(len(x)):
        if(x[i]>1):
            x[i]=1
    
    return x

def projectBack_ball(x):
    nrm = np.linalg.norm(x)
    if(nrm>1):
        x = x / nrm;
    return x

# In[]
def computeGrad(w,x,y):
    y_p = np.dot(x,w.T)
    t = np.exp(-1.0 * y_p * y)
    return - y* x * t / (1+t)

def updateW(w, X, y, a_t):
    return w - a_t * computeGrad(w,X,y)

def calculateWHat(W):
    return np.mean(W)

def loss(w,X,y):
    y_p = np.dot(x,w.T)
    t = np.exp(-1.0 * y_p * y)
    return np.log(1+t)
        
def error(w,X,y):
    y_p = np.dot(x,w.T)
    return np.sign(y)==np.sign(y_p)

def oracle():
    global sample_no
    assert(sample_no < n_train+n_test)
    z = dist_X[sample_no]
    y = dist_y[sample_no]
    sample_no += 1
    
    return z, y

def project(w, dist_type):
    if dist_type == "box":
        if isOutOfSet_hypercube(w):
            return projectBack_hypercube(w)
        
    elif dist_type =="ball":
        if isOutOfSet_ball(w):
            return projectBack_ball(w)
        
    return w
def getAlpha(dist_type):
    if dist_type == "box":
        return np.sqrt(dim_C)/(np.sqrt(dim_C)*np.sqrt(T))
    elif dist_type =="ball":
        return 1.0/(np.sqrt(2)*np.sqrt(T))
    return 0
# In[]
def SGD_step(w_t, alpha_t):
    z,y = oracle()
    grad = computeGrad(w_t,z,y)
    w_t_1 = w_t - alpha_t * grad
    w_t_1 = project(w_t_1, dist_type)
    return w_t_1


def SGD():
    global sample_no
    sample_no = 0
    alpha = getAlpha(dist_type)
    Wt_s = []
    w = np.zeros((dim_C,1))
    Wt_s.append(w)
    for i in range(n_train):
        w = SGD_step(w,alpha)
        Wt_s.append(w)
    w_hat = calculateWHat(Wt_s)
    return w_hat