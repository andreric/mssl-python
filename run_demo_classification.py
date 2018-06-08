#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 17:06:57 2018

@author: goncalves1
"""

import pickle
import numpy as np
from mssl.MSSLClassifier import MSSLClassifier


# Classification
print('Loading data ...')
with open('datasets/toy_10tasks_clf.pkl', 'rb') as fh:
    x, y, dimension, ntasks = pickle.load(fh)

nsamples = x[0].shape[0]

# optimization algorithm for W-step
#   options: 'closed-form', 'fminunc', 'lbfgs' and 'fista'
#
#opts.alg_w = 'closed-form'

# optimization algorithm for Omega-step (structure learning)
#opts.alg_theta = 'admm'  # only ADMM available so far

rho_1 = 0.28  # controls sparsity on the Omega matrix (tasks' relationship)
rho_2 = 0.5  # controls sparsity on tasks' parameter matrix W (only for FISTA)

#opts.maxiter = 50     # maximum number of iterations for the alternating minimization algorithm
res_mssl = list()

# number of runs
for k in range(5):
    
     # selecting data for training
    ids = np.random.choice(np.arange(nsamples), size=40, replace=False)
    rid = np.setdiff1d(np.arange(nsamples), ids)
    
    nid = np.random.choice(np.arange(len(rid)), size=60, replace=False)
    ntasks = 3
    xtrain = [None]*ntasks
    ytrain = [None]*ntasks
    xtest = [None]*ntasks
    ytest = [None]*ntasks
    
    # splitting data in training and test
    for i in range(ntasks):
        
        xtrain[i] = x[i][ids,:]
        ytrain[i] = y[i][ids]

        # normalization (indicated when data is not gaussian)
        # xtrain{i} = zscore(xtrain{i})
        # [ytrain{i},mu{i},sigma{i}] = zscore( ytrain{i} ) # preprocessing data - Standartization: x ~ N(0,1)

        xtest[i] = x[i][rid[nid], :]
        ytest[i] = y[i][rid[nid]]


    mssl_clf = MSSLClassifier(lambda_1=rho_1, lambda_2=rho_2)
    mssl_clf.fit(xtrain, ytrain)
    W_mssl = mssl_clf.W
    Theta = mssl_clf.Omega
    
    yhat = mssl_clf.predict(xtest)
    
    acc = [[] for _ in range(ntasks)]
    for t in range(ntasks):
        acc[t] = (yhat[t] == ytest[t]).sum()/len(ytest[t])
    print(acc)

    ## Testing phase - and - Performance evaluation
    res_mssl.append({'W': mssl_clf.W, 'Omega': mssl_clf.Omega})

#     ## Single task learning - STL (tasks are learnt independently)
#     [W_stl] = ls_independents( xtrain, ytrain, k )
#     res_stl{k} = perf_regression( xtest, ytest, [], W_stl, [], [] )
#     res_stl{k}.w = W_stl
#     
#     rmse_mssl(k,:) = res_mssl{k}.rmse
#     rmse_ls(k,:) = res_stl{k}.rmse

#        
#x = rmse_mssl  # rmse obtained by p-MSSL
#y = rmse_ls    # rmse obtained by STL
#
## plot performance (rmse) for all tasks
#h = cat(1, reshape(x,[1 size(x)]), reshape(y,[1 size(y)]))
#aboxplot(h,'labels',1:10) # Advanced box plot
#xlabel('Tasks','fontsize',17)
#ylabel('RMSE','fontsize',17)
#    
#legend('p-MSSL','OLS')
#
#save('mssl_output.mat','rmse_mssl','rmse_ls')
#fprintf('\nResults were saved in ''mssl_output.mat'' file.\n\n')

  
