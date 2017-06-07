# -*- coding: utf-8 -*-
"""
CSE 250C HW 3

"""
from __future__ import print_function
            
# In[]
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(10)
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
        if(np.abs(x[i])>1):
            x[i]=1.0 * np.sign(x[i])
    
    return x

def projectBack_ball(x):
    nrm = np.linalg.norm(x)
    if(nrm>1):
        x = x / nrm;
    return x

def project(w, dist_type):
    assert(type(dist_type)==str)
    w = w.reshape((-1,1))
    if dist_type == "hypercube":
        if isOutOfSet_hypercube(w):
            return True,projectBack_hypercube(w)
        
    elif dist_type =="ball":
        if isOutOfSet_ball(w):
            return True,projectBack_ball(w)
    w = w.reshape((1,-1))
    return False,w

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

def getMRho(dim,dist_type):
    assert(type(dist_type)==str)
    
    if dist_type =="hypercube":
        return {"M":np.sqrt(dim), "rho": np.sqrt(dim)}
    else:
        return {"M":1.0, "rho": np.sqrt(2.0)}

def getAlpha(dim,T,dist_type):
    # alpha = M / (rho * sqrt(T))
    M = getMRho(dim,dist_type)["M"]
    rho = getMRho(dim,dist_type)["rho"]
    
    return M / (rho * np.sqrt(T))
    
def oracle(mean, var, dim, dist_type):
    y = 1.0 if (np.random.uniform(0.0,1.0,1) >= 0.5) else -1.0
    cov = np.identity(dim) * var
    mean = [mean*y]*dim
    z = np.random.multivariate_normal(mean, cov, 1)
    isout,z = project(z, dist_type)
    z = np.insert(z,0,1)
    return z, y,isout

# In[]
def SGD_step(w_t, alpha_t, z, y, dist_type):
    grad = computeGrad(w_t,z,y)
    w_t_1 = w_t - alpha_t * grad
    isout,w_t_1 = project(w_t_1, dist_type)
    return w_t_1.reshape((1,-1)),isout


def SGD(mean, var, dim_X, dim_C, T, dist_type):
    
    alpha = getAlpha(dim_C, T, dist_type)
    Wt_s = []
    w = np.zeros((1,dim_C))
    Wt_s.append(w)
    data = []
    isOut = []
    isOut_w = []
    for i in range(1,T,1):
        z,y,isout = oracle(mean, var, dim_X, dist_type)
        isOut += [isout]
        w,isout = SGD_step(w, alpha, z, y, dist_type)
        isOut_w += [isout]
        Wt_s.append(w)
        data.append((z,y))
        #print(w)
    w_hat = calculateWHat(Wt_s)
    return w_hat, Wt_s, data, np.mean(isOut), np.mean(isOut_w)

# In[]
def getTestData(mean, var, dim, size, dist_type):
    data = []
    count = 0
    for i in range(size):
        z,y,isout = oracle(mean, var, dim, dist_type)
        if isout:
            count += 1 
        data.append((z,y))
    print(dist_type,"test data out =",float(count)/float(size));
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
    dim_C = dim_X + 1
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
        plt.figure()
        two_dim_dat = np.array([x[0][1:3] for x in data_test if x[1]==1])
        plt.scatter(two_dim_dat[:,0], two_dim_dat[:,1])
        two_dim_dat = np.array([x[0][1:3] for x in data_test if x[1]==-1])
        plt.scatter(two_dim_dat[:,0], two_dim_dat[:,1])
        plt.title("Scatter plot for "+dist_type+" with std="+str(np.sqrt(var)))
        plt.savefig("sgd/scatter_"+dist_type+"_"+str(np.sqrt(var)).replace('.','_'))
        
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
            isOut = []
            isOut_w = []
            #w_hat_const = []
            for _i in range(num_iter):
                w_hat, Wt_s, data_train, is_out,is_out_w = SGD(mean, var, dim_X, dim_C, T, dist_type)
                isOut += [is_out]
                isOut_w += [is_out_w]
                #expLoss_train  += [expectedLoss(data_train, w_hat)]
                #expError_train += [expectedError(data_train, w_hat)]
                expLoss_test   += [expectedLoss(data_test, w_hat)]
                expError_test  += [expectedError(data_test, w_hat)]
                if _i==0:
                    print (w_hat, np.linalg.norm(w_hat))
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
            
            scen = 1 if dist_type=="hypercube" else 2 
            
            print ("expected_excess_risk_loss = ",expected_excess_risk_loss)
            print ("expected_avg_class_error = ",expected_avg_class_error)
            if True:
                file.write(('{0:1d} & {1:.2f} & {2:12d} & {3:12d} & {4:12d}'
                         ' & {5:.1f} & {6:.1f}\\\\ \n').format(\
                         scen, np.sqrt(var),n_train, n_test,30,\
                         np.mean(isOut)*100.0, np.mean(isOut_w)*100.0))
            if False:
                file.write(('{0:1d} & {1:.2f} & {2:12d} & {3:12d} & {4:12d}'
                         ' & {5:.3g} & {6:.3g} & {7:.3g} & {8:.3g}'
                         ' & {9:.3g} & {10:.3g} \\\\ \n').format(\
                         scen, np.sqrt(var),n_train, n_test,30,\
                         avg_risk_loss, std_risk_loss, min_risk_loss,expected_excess_risk_loss,\
                         avg_class_error,std_class_error))
            if False:                
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
# In[]
import os
if not os.path.isdir("sgd"):
    os.mkdir("sgd")

# In[]
np.random.seed(10)
dist_type_arr = ["hypercube","ball"]
filename="sgd/HW3_results_out.txt"
f=open(filename,'w')
risk_error_hypercube = analysis(dist_type_arr[0],f)
risk_error_ball = analysis(dist_type_arr[1],f)
f.close()
# In[]
if False:
    np.save("sgd/risk_error_hypercube",risk_error_hypercube)
    np.save("sgd/risk_error_ball",risk_error_ball)


def theoriticalBounds(dim,dist_type):
    #n_arr = list(range(50,1001,50))
    n_arr = [50, 100, 500, 1000]
    M = getMRho(dim,dist_type)["M"]
    rho = getMRho(dim,dist_type)["rho"]
    print(M,rho)
    L = []
    for n in n_arr:
        T = n+1
        L += [M*rho/np.sqrt(T)]
    return n_arr, L

fig, ax = plt.subplots()
n,theo_loss = theoriticalBounds(6,dist_type_arr[0])
ax.plot(n,np.array(theo_loss))
#ax.plot(n,np.array(theo_loss)/60.0, label="theorical-2")
for i, txt in enumerate(np.round(theo_loss,3)):
    ax.annotate(txt, (n[i],theo_loss[i]))
plt.legend()
plt.ylabel("expected risk")
plt.xlabel("#training_samples")
plt.title("Hypercube: Theoretical upperbound on \nexpected risk vs #training_samples")
plt.savefig("sgd/hypercube_exp_theo_risk",bbox_inches='tight')

fig, ax = plt.subplots()
n,theo_loss = theoriticalBounds(6,dist_type_arr[1])
ax.plot(n,theo_loss, label=dist_type_arr[1])
for i, txt in enumerate(np.round(theo_loss,3)):
    ax.annotate(txt, (n[i],theo_loss[i]))
plt.xticks(n)
plt.ylabel("expected risk")
plt.xlabel("#training_samples")
#plt.legend()
plt.title("Ball: Theoretical upperbound on \nexpected risk vs #training_samples")
#plt.savefig("sgd/ball_exp_theo_risk",bbox_inches='tight')
plt.figure()
# In[]
def theoriticalBounds(dim,dist_type):
    #n_arr = list(range(50,1001,50))
    n_arr = [50, 100, 500, 1000]
    M = getMRho(dim,dist_type)["M"]
    rho = getMRho(dim,dist_type)["rho"]
    print(M,rho)
    L = []
    for n in n_arr:
        T = n+1
        L += [M*rho/np.sqrt(T)]
    return n_arr, L

fig, ax = plt.subplots()
n,theo_loss = theoriticalBounds(6,dist_type_arr[0])
ax.plot(n,np.array(theo_loss), label="theoretical")
plt.yscale("log")
#ax.plot(n,np.array(theo_loss)/60.0, label="theorical-2")
for i, txt in enumerate(np.round(theo_loss,3)):
    ax.annotate(txt, (n[i],theo_loss[i]))
for std, risk_error in risk_error_hypercube.items():
    ax.errorbar(risk_error["n"], risk_error["risk"], yerr=risk_error["risk_std"], label="std="+str(std), capthick=1, capsize=3, barsabove=True)

plt.legend()
plt.ylabel("expected risk")
plt.xlabel("#training_samples")
plt.title("Hypercube: Comparision with theorical upper bound")
plt.savefig("sgd/hypercube_exp_theo_risk_cmp",bbox_inches='tight')

fig, ax = plt.subplots()
n,theo_loss = theoriticalBounds(6,dist_type_arr[1])
ax.plot(n,np.array(theo_loss), label="theoretical")
for i, txt in enumerate(np.round(theo_loss,3)):
    ax.annotate(txt, (n[i],theo_loss[i]))
for std, risk_error in risk_error_ball.items():
    ax.errorbar(risk_error["n"], risk_error["risk"], yerr=risk_error["risk_std"], label="std="+str(std), capthick=1, capsize=3, barsabove=True)
plt.xticks(n)
plt.yscale("log")
plt.legend()
plt.ylabel("expected risk")
plt.xlabel("#training_samples")
#plt.legend()
plt.title("Ball: Comparision with theorical upper bound")
plt.savefig("sgd/ball_exp_theo_risk_cmp",bbox_inches='tight')
plt.figure()


# In[]
fig, ax = plt.subplots()
    
for std, risk_error in risk_error_hypercube.items():
    ax.errorbar(risk_error["n"], risk_error["risk"], yerr=risk_error["risk_std"], label="std="+str(std), capthick=1, capsize=3, barsabove=True)
    plt.ylabel("expected excess risk")
    plt.xlabel("#training_samples")
    plt.xticks(n)
    plt.legend()
    plt.title("Hypercube: expected excess risk vs #training_samples")
    plt.savefig("sgd/hypercube_exp_risk",bbox_inches='tight')

fig, ax = plt.subplots()
for std, risk_error in risk_error_hypercube.items():
    ax.errorbar(risk_error["n"], risk_error["error"], yerr=risk_error["error_std"], label="std="+str(std), capthick=1, capsize=3, barsabove=True)
    plt.ylabel("expected error")
    plt.xlabel("#training_samples")
    plt.legend()
    plt.xticks(n)
    plt.title("Hypercube: expected classification error vs #training_samples")
    plt.savefig("sgd/hypercube_exp_error",bbox_inches='tight')

fig, ax = plt.subplots()
for std, risk_error in risk_error_ball.items():
    ax.errorbar(risk_error["n"], risk_error["risk"], yerr=risk_error["risk_std"], label="std="+str(std), capthick=1, capsize=3, barsabove=True)
    plt.ylabel("expected excess risk")
    plt.xlabel("#training_samples")
    plt.legend()
    plt.xticks(n)
    plt.title("Ball: expected excess risk vs #training_samples")
    plt.savefig("sgd/ball_exp_risk",bbox_inches='tight')
    
fig, ax = plt.subplots()
for std, risk_error in risk_error_ball.items():
    ax.errorbar(risk_error["n"], risk_error["error"], yerr=risk_error["error_std"], label="std="+str(std), capthick=1, capsize=3, barsabove=True)
    plt.ylabel("expected error")
    plt.xlabel("#training_samples")
    plt.legend()
    plt.xticks(n)
    plt.title("Ball: expected classification error vs #training_samples")
    plt.savefig("sgd/ball_exp_error",bbox_inches='tight')


plt.show()
    
