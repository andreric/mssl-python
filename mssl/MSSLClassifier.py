#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:27:34 2018

@author: goncalves1
"""
import sys
import os
import pickle
import numpy as np
import scipy.special
import scipy.optimize

#sys.path.append('..')
#from design import Method


def logloss(w, x, y, Omega, P):  #x_vec, y_vec, theta, perm_matrix):
#    print(type(args))
#    x, y, Omega, P = args['x'], args['y'], args['Omega'], args['P']

    ntasks = Omega.shape[1]
    ndimension = int(len(w)/ntasks)

    # cost function
    a = np.dot(x, w)
    a = np.maximum(np.minimum(a, 50), -50)
    f = -(y*(a-np.log(1+np.exp(a))) + (1-y)*(-np.log(1+np.exp(a)))).sum()

    r = np.reshape(w, (ndimension, ntasks), order='F')
#    r = w.reshape(ndimension, ntasks, order='F').copy()
    f += 0.5*np.trace(np.dot(np.dot(r, Omega), r.T))
    print(f)

    # gradient of the cost function
    # logistic regression term
#    sig_s = scipy.special.expit(np.dot(x,w))[:, np.newaxis]
    sig_s = sigmoid(np.dot(x,w))[:, np.newaxis]
    g0 = np.dot(x.T, (sig_s-y))
    matm = np.kron(Omega, np.eye(ndimension))

    # regularization term
    g1 = np.dot(np.dot(np.dot(P, matm), P.T), w)
    g = g0[:, 0] + g1

    return f, g

def sigmoid(a):
    # Logit function for logistic regression
    # x: data point (array or a matrix of floats)
    # w: set of weights (array of float)

    # treating over/underflow problems
    a = np.maximum(np.minimum(a, 50), -50)
    #f = 1./(1+exp(-a));
    f = np.exp(a) / (1+np.exp(a))
    return f

class MSSLClassifier():
    """
    Implement the MSSL classifier.

    Attributes:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
    """
    def __init__(self, rho_1=0.1, rho_2=0):
        """ Initialize object with the informed hyper-parameter values.
        Args:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
        """
        # set method's name and paradigm
#        super().__init__('MSSLClassifier', 'MTL')

        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.max_iters = 100
        self.tol = 1e-5  # minimum tolerance: eps * 100

        self.admm_rho = 1  # ADMM parameter
        self.eps_theta = 1e-3   # stopping criteria parameters
        self.eps_w = 1e-3    # stopping criteria parameters

        self.ntasks = -1
        self.dimension = -1
        self.W = None
        self.Omega = None
        self.output_directory = ''
#        self.__create_permutation_matrix = __create_permutation_matrix
#        self.__create_sparse_AC = __create_sparse_AC

    def fit(self, x, y):

        print('\n[MSSL]')

        # get number of tasks
        self.ntasks = len(x)
        self.dimension = x[0].shape[1]

        # create permutation matrix
        P = self.__create_permutation_matrix(self.dimension, self.ntasks)
#        P = self.__create_permutation_matrix(2, 3)


        # create (sparse) matrices x and y that are the concatenation
        # of data from all tasks. So, in this way we convert multiple
        # tasks problem to a single (bigger) problem used in the
        # closed-form solution. For gradient-based algorithms like Fista,
        # L-BFGS and others, this is not needed (although can also be used)
        xmat, ymat = self.__create_sparse_AC(x, y)

        # Learning parameters
        Wvec = np.random.rand(self.ntasks*self.dimension, 1) #np.linalg.solve(xmat, ymat)  #  warm start
#        print(Wvec.shape)
        Omega = np.eye(self.ntasks)

        # Limited memory BFGS
#        if self.alg_w == 'lbfgs':,
#            opt_fminunc = struct('factr', 1e4, 'pgtol', 1e-5, 'm', 10,'maxIts',10,'printEvery',1e6);
#            if ntasks*dimension > 10000:
#                opt_fminunc.m  = 50

        opts = {'maxiter': 1000, 'disp':True}

        for it in range(self.max_iters):
            print('%d ' % it)

            # Minimization step
            W_old = Wvec.copy()

            additional = (xmat, ymat, Omega, P)
            res = scipy.optimize.minimize(logloss, x0=Wvec,
                                          args=additional,
                                          jac=True, method='BFGS',
                                          options=opts)
#            print(res.status)
            Wvec = res.x.copy()

#            # put it in matrix format, where columns are coeff for each task
            Wmat = np.reshape(Wvec, (self.dimension, self.ntasks), order='F')
            # Omega step:
            Omega_old = Omega
#
#            # Learn relationship between tasks (inverse covariance matrix)
##            Omega = __covsel_admm(np.cov(Wmat), self.rho2, self.rho)
#
#            # checking convergence of Omega and W
            diff_Omega = np.linalg.norm(Omega-Omega_old)
            diff_W = np.linalg.norm(Wvec-W_old)
#
#            # if difference between two consecutive iterations are very small,
#            # stop training
            if (diff_Omega < self.eps_theta) and (diff_W < self.eps_w):
                break
        self.W = Wmat.copy()
        self.Omega = Omega.copy()

    def predict(self, x):
        yhat = [None]*len(x)
        for t in range(len(x)):
            yhat[t] = scipy.special.expit(np.dot( x[t], self.W[:, t]))
            yhat[t] =  np.round(yhat[t]).astype(np.int32)
        return yhat

    def __create_permutation_matrix(self, k, v):
        I = np.eye(k*v)
        permV = np.zeros((k*v+1, 1),dtype=int)
        cur = 1
        for i in range(1,v+1):
            for j in range(1,k+1):
                permV[cur,0] = i+((j-1)*v)
                cur = cur+1
        permV = permV[1:]-1
        p = I[:, permV].T
        return np.squeeze(p)
    
    def __create_sparse_AC(self, A, b):
        nmodels = A[0].shape[1]
        ntasks = len(A)
        Acap = np.zeros((nmodels*ntasks,1))
        C = np.zeros((nmodels*ntasks,1))

        beginId = 0
        endId = nmodels
        for task in range(ntasks):
            splI = np.zeros((ntasks, ntasks))
            splI[task][task] = 1 #set this node to 1 so it gets picked up
            #do kronecker product and add to Acap
            newA = np.dot(A[task].T, A[task])
            Acap = Acap + np.kron(splI, newA)
            C[beginId:endId] = np.dot(A[task].T, b[task])
            beginId = endId
            endId = endId+nmodels
        return Acap, C
        






def __covsel_admm():
    pass

#function [Z,f] = covsel_admm( S, lambda, rho )
# covsel  Sparse inverse covariance selection via ADMM
#
# [X, history] = covsel(D, lambda, rho, alpha)
#
# Solves the following problem via ADMM:
#
#   minimize  trace(S*X) - log det X + lambda*||X||_1
#
# with variable X, where S is the empirical covariance of the data
# matrix D (training observations by features).
#
# The solution is returned in the matrix X.
#
# history is a structure that contains the objective value, the primal and
# dual residual norms, and the tolerances for the primal and dual residual
# norms at each iteration.
#
# rho is the augmented Lagrangian parameter.
#
# alpha is the over-relaxation parameter (typical values for alpha are
# between 1.0 and 1.8).
#
# More information can be found in the paper linked at:
# http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
#

#    #Global constants and defaults
#
#    MAX_ITER = 500;
#    ABSTOL   = 1e-5;
#    RELTOL   = 1e-5;
#    ALFA     = 1.4;
#    #Data preprocessing
#
#    n = size(S,1);
#    #ADMM solver
#
#    #X = zeros(n);
#    Z = zeros(n);
#    U = zeros(n);
#
#    #if ~QUIET
#    #    fprintf('#3s\t#10s\t#10s\t%10s\t%10s\t%10s\n', 'iter', ...
#    #      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
#    #end
#
#    #dimension = size(D,1);
#
#    for k = 1:MAX_ITER
#
#        # x-update
#        #[Q,L] = eig(rho*(Z - U) - (S/2));
#        [Q,L] = eig(rho*(Z - U) - S);
#        es = diag(L);
#        #xi = (es + sqrt(es.^2 + 2*rho*dimension))./(2*rho);
#        xi = (es + sqrt(es.^2 + 4*rho))./(2*rho);
#        X = Q*diag(xi)*Q';
#
#        # z-update with relaxation
#        Zold = Z;
#        X_hat = ALFA*X + (1 - ALFA)*Zold;
#        Z = shrinkage(X_hat + U, lambda/rho);
#
#        U = U + (X_hat - Z);
#
#        # diagnostics, reporting, termination checks
#
#        #objval(k)  = objective_function(S, X, Z, dimension, lambda);
#
#        r_norm  = norm(X - Z, 'fro');
#        s_norm  = norm(-rho*(Z - Zold),'fro');
#
#        eps_pri = sqrt(n*n)*ABSTOL + RELTOL*max(norm(X,'fro'), norm(Z,'fro'));
#        eps_dual= sqrt(n*n)*ABSTOL + RELTOL*norm(rho*U,'fro');
#
#        if (r_norm < eps_pri) and (s_norm < eps_dual):
#             break
#
#        f = objective_function( S, X, lambda )
#
#
#    def __objective_function(S, X, rho):
#        obj = 0.5*trace(S*X) - log(det(X)) + rho*norm(X(:), 1)
#        return obj
#
#    def __shrinkage(a, kappa):
#        return np.maximum(0, a-kappa) - np.maximum(0, -a-kappa)
