# -*- coding: utf-8 -*-
"""
CSE 250C HW 3

"""
# In[]
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
# In[]
'''
def getDist(var, size):
    dim_X = 5
    mean = 1
    mean_n = [-mean]*dim_X
    mean_p = [mean]*dim_X
    ones = np.random.uniform(0.0,1.0,size) > 0.5
    y = np.zeros(size);
    y[ones] = 1
    y[True^ones] = -1
    X = np.zeros((size,dim_X))
    cov = np.identity(dim_X) * var
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

#In[]
def plotDist(X, a=1):
    sns.set(color_codes=True)
    sns.distplot(X[:,a]);

#In[]

dist_X,dist_y = getDist(1,1000);
#X_test,y_test   = getDist(var,n_test);
plt.plot(dist_X[:,:2],'.')
plt.show()
'''
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

def project(w, dist_type):
    w = w.reshape((-1,1))
    if dist_type == "box":
        if isOutOfSet_hypercube(w):
            return projectBack_hypercube(w)
        
    elif dist_type =="ball":
        if isOutOfSet_ball(w):
            return projectBack_ball(w)
    w = w.reshape((1,-1))
    return w

# In[]
def computeGrad(w,x,y):
    y_p = np.dot(x,w.T)
    t = np.exp(-1.0 * y_p * y)
    return - y* x * t / (1+t)

def loss(w,X,y):
    y_p = np.dot(X,w.T)
    t = np.exp(-1.0 * y_p * y)
    return np.log(1+t)
        
def error(w,X,y):
    y_p = np.dot(X,w.T)
    return np.sign(y)!=np.sign(y_p)
# In[]
def calculateWHat(W):
    return np.mean(W, axis=0).reshape((1,-1))

def getAlpha(dim,T,dist_type):
    # alpha = M / (rho * sqrt(T))
    if dist_type == "box":
        return (1.0*np.sqrt(dim))/(np.sqrt(dim)*np.sqrt(T))
    elif dist_type =="ball":
        return 1.0/(np.sqrt(2)*np.sqrt(T))
    return 0

def oracle(mean, var, dim, dist_type):
    y = 1.0 if (np.random.uniform(0.0,1.0,1) >= 0.5) else -1.0
    cov = np.identity(dim) * var
    mean = [mean*y]*dim
    z = np.random.multivariate_normal(mean, cov, 1)
    z = project(z, dist_type)
    z = np.insert(z,0,1)
    return z, y

# In[]
def SGD_step(w_t, alpha_t, z, y, dist_type):
    grad = computeGrad(w_t,z,y)
    w_t_1 = w_t - alpha_t * grad
    w_t_1 = project(w_t_1, dist_type)
    return w_t_1.reshape((1,-1))


def SGD(mean, var, dim_X, dim_C, T, dist_type):
    
    alpha = getAlpha(dim_C, T, dist_type)
    Wt_s = []
    w = np.zeros((1,dim_C))
    Wt_s.append(w)
    data = []
    for i in range(1,T,1):
        z,y = oracle(mean, var, dim_X, dist_type)
        w = SGD_step(w, alpha, z, y, dist_type)
        Wt_s.append(w)
        data.append((z,y))
        #print(w)
    w_hat = calculateWHat(Wt_s)
    return w_hat, Wt_s, data

# In[]
def getTestData(mean, var, dim, size, dist_type):
    data = []
    for i in range(size):
        z,y = oracle(mean, var, dim, dist_type)
        data.append((z,y))
    return data

def expectedLoss(data, w_hat):
    _loss = 0.0
    for (X,y) in data:
        _loss += loss(w_hat, X, y)
    
    return _loss / float(len(data))

def expectedError(data, w_hat):
    _error = 0.0
    for (X,y) in data:
        _error += 1.0 if error(w_hat, X, y) else 0.0
    
    return _error / float(len(data))

def calulate_min_avg_std(X):
    return np.min(X), np.mean(X), np.std(X)

# In[]
def analysis(dist_type,file):
    print( "analysis for ",dist_type)
 

    dim_X = 5
    dim_C = 6
    n_train_arr = [50, 100, 500, 1000]
    n_test = 400
            
    std = [0.05, 0.3]
    #std= np.linspace(0.05, 0.3, 11, True)
    var_arr = np.square(std)
    #num_iter = 30
    num_iter = 30
    mean = 1.0/5.0
    risk_error = dict()
    for var in var_arr:
        print ("var=",var)
        data_test = getTestData(mean, var, dim_X, n_test, dist_type)
        exp_excess_risk = []
        exp_excess_risks_std = []
        exp_class_mean = []
        exp_class_std  = []
        
        for n_train in n_train_arr:
            print ("n_train=",n_train)
            T = n_train + 1
            #expLoss_train = []
            #expError_train = []
            expLoss_test = []
            expError_test = []
            for _i in range(num_iter):
                w_hat, Wt_s, data_train = SGD(mean, var, dim_X, dim_C, T, dist_type)
                #expLoss_train  += [expectedLoss(data_train, w_hat)]
                #expError_train += [expectedError(data_train, w_hat)]
                expLoss_test   += [expectedLoss(data_test, w_hat)]
                expError_test  += [expectedError(data_test, w_hat)]
            
            # risk and error calculations
            min_risk_loss, avg_risk_loss, std_risk_loss \
                            = calulate_min_avg_std(expLoss_test)
            
            min_class_error, avg_class_error, std_class_error \
                            = calulate_min_avg_std(expError_test)
            
            expected_excess_risk_loss = avg_risk_loss - min_risk_loss
            expected_avg_class_error = avg_class_error
            
            exp_excess_risk += [expected_excess_risk_loss]
            exp_excess_risks_std += [std_risk_loss]
            exp_class_mean += [avg_class_error]
            exp_class_std += [std_class_error]
            
            scen = 1 if dist_type=="box" else 2 
            
            print ("expected_excess_risk_loss = ",expected_excess_risk_loss)
            print ("expected_avg_class_error = ",expected_avg_class_error)
            file.write(('{0:1d} & {1:.2f} & {2:12d} & {3:12d} & {4:12d}'
                     ' & {5:.3e} & {6:.3e} & {7:.3e} & {8:.3e}'
                     ' & {9:.3e} & {10:.3e} \\\\ \n').format(\
                     scen, np.sqrt(var),n_train, n_test,30,\
                     avg_risk_loss, std_risk_loss, min_risk_loss,expected_excess_risk_loss,\
                     avg_class_error,std_class_error))
        file.write("\hline\n")
        
        risk_error[np.sqrt(var)] = {"risk":exp_excess_risk, "risk_std":exp_excess_risks_std, \
                   "error":exp_class_mean, "error_std":exp_class_std, "n": n_train_arr}
    return risk_error
# In[]
#if (__name__=="__main__"): 
dist_type = ["box","ball"]
filename="./HW3_results.txt_temp"
f=open(filename,'w')
risk_error_box = analysis(dist_type[0],f)
risk_error_ball = analysis(dist_type[1],f)
f.close()

# In[]
for std, risk_error in risk_error_ball.items():
    fig, ax = plt.subplots()
    ax.errorbar(risk_error["n"], risk_error["risk"], yerr=risk_error["risk_std"])
    plt.ylabel("expected risk")
    plt.xlabel("n")
    plt.title("expected risk vs n for std="+str(std))
    plt.savefig("exp_risk_"+str(std).replace('.','_'))
    
    fig, ax = plt.subplots()
    ax.errorbar(risk_error["n"], risk_error["error"], yerr=risk_error["error_std"])
    plt.ylabel("expected error")
    plt.xlabel("n")
    plt.title("expected classification error vs n for std="+str(std))
    plt.savefig("exp_error_"+str(std).replace('.','_'))
    
    plt.show()


    
    
