#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 17:06:57 2018

@author: goncalves1
"""

from MSSL import MSSLClassifier

#addpath('alternating/')
#addpath('single_task_learning')
#addpath('performance_analysis')
#addpath('aboxplot/')
#addpath('datasets')


# Regression
print('Loading data ...')
load toy_10tasks.mat 

# optimization algorithm for W-step
#   options: 'closed-form', 'fminunc', 'lbfgs' and 'fista'
#
#opts.alg_w = 'closed-form'

# optimization algorithm for Omega-step (structure learning)
#opts.alg_theta = 'admm'  # only ADMM available so far

rho_1 = 0.28  # controls sparsity on the Omega matrix (tasks' relationship)
rho_2 = 0.5  # controls sparsity on tasks' parameter matrix W (only for FISTA)

#opts.maxiter = 50     # maximum number of iterations for the alternating minimization algorithm

# number of runs
for k in range(5):
    
     # selecting data for training
     id = randperm(300,40)
     rid = setdiff(1:300, id)
     
     nid = randperm( length(rid), 60 )
     
     # splitting data in training and test
     for i in range(ntasks):
         
        xtrain[i] = x[i][id,:]
        ytrain[i] = y[i][id]
        
        # normalization (indicated when data is not gaussian)
        # xtrain{i} = zscore(xtrain{i})               
        # [ytrain{i},mu{i},sigma{i}] = zscore( ytrain{i} ) # preprocessing data - Standartization: x ~ N(0,1)
    
        xtest{i} = x{i}(rid(nid),:)
        ytest{i} = y{i}(rid(nid))
        
     end

     [W_mssl,Theta] = alternating_minimization( 'regression', xtrain, ytrain, opts )
     
     ## Testing phase - and - Performance evaluation
     res_mssl{k} = perf_regression( xtest, ytest, [], W_mssl, [], [] )
     res_mssl{k}.w = W_mssl
     res_mssl{k}.theta = Theta
    
     
     ## Single task learning - STL (tasks are learnt independently)
     [W_stl] = ls_independents( xtrain, ytrain, k )
     res_stl{k} = perf_regression( xtest, ytest, [], W_stl, [], [] )
     res_stl{k}.w = W_stl
     
     rmse_mssl(k,:) = res_mssl{k}.rmse
     rmse_ls(k,:) = res_stl{k}.rmse
     
end
        
x = rmse_mssl  # rmse obtained by p-MSSL
y = rmse_ls    # rmse obtained by STL

# plot performance (rmse) for all tasks
h = cat(1, reshape(x,[1 size(x)]), reshape(y,[1 size(y)]))
aboxplot(h,'labels',1:10) # Advanced box plot
xlabel('Tasks','fontsize',17)
ylabel('RMSE','fontsize',17)
    
legend('p-MSSL','OLS')

save('mssl_output.mat','rmse_mssl','rmse_ls')
fprintf('\nResults were saved in ''mssl_output.mat'' file.\n\n')

  
