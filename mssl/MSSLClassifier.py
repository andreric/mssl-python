#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:27:34 2018

@author: goncalves1
"""
#import sys
#import os
#import pickle
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
    a = np.maximum(np.minimum(np.dot(x, w), 50), -50)
    f1 = np.multiply(y, a-np.log(1+np.exp(a)))
    f2 = np.multiply(1-y, -np.log(1+np.exp(a)))
    f = -(f1 + f2).sum()

    r = np.reshape(w, (ndimension, ntasks), order='F')
#    r = w.reshape(ndimension, ntasks, order='F').copy()
    f += 0.5*np.trace(np.dot(np.dot(r, Omega), r.T))
#    print(f)

    # gradient of the cost function
    # logistic regression term
#    sig_s = scipy.special.expit(np.dot(x,w))[:, np.newaxis]
    sig_s = sigmoid(np.dot(x, w))[:, np.newaxis]
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


def shrinkage(a, kappa):
    return np.maximum(0, a-kappa) - np.maximum(0, -a-kappa)


class MSSLClassifier():
    """
    Implement the MSSL classifier.

    Attributes:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
    """
    def __init__(self, lambda_1=0.1, lambda_2=0):
        """ Initialize object with the informed hyper-parameter values.
        Args:
        rho_L21 (float): l2,1 penalization hyper-parameter
        rho_L2 (float): l2 penalization hyper-parameter
        """
        # set method's name and paradigm
#        super().__init__('MSSLClassifier', 'MTL')

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
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

        # get number of tasks
        self.ntasks = len(x)
        self.dimension = x[0].shape[1]

        # create permutation matrix
        P = self.create_permutation_matrix(self.dimension, self.ntasks)

        # create (sparse) matrices x and y that are the concatenation
        # of data from all tasks. So, in this way we convert multiple
        # tasks problem to a single (bigger) problem used in the
        # closed-form solution. For gradient-based algorithms like Fista,
        # L-BFGS and others, this is not needed (although can also be used)
        xmat, ymat = self.create_sparse_AC(x, y)

        # initialize learning parameters
        # np.linalg.solve(xmat, ymat)  #  regression warm start
        Wvec = -0.05 + 0.1*np.random.rand(self.ntasks*self.dimension, 1)
        Omega = np.eye(self.ntasks)

        # Limited memory BFGS
#        if self.alg_w == 'lbfgs':,
#            opt_fminunc = struct('factr', 1e4, 'pgtol', 1e-5, 'm', 10,'maxIts',10,'printEvery',1e6);
#            if ntasks*dimension > 10000:
#                opt_fminunc.m  = 50

        # scipy opt parameters
        opts = {'maxiter': 10, 'disp':True}
        for it in range(self.max_iters):
            print('%d ' % it)

            # Minimization step
            W_old = Wvec.copy()

            additional = (xmat, ymat, Omega, P)
            res = scipy.optimize.minimize(logloss, x0=Wvec,
                                          args=additional,
                                          jac=False, method='BFGS',
                                          options=opts)
#            print(res.status)
            Wvec = res.x.copy()

#            # put it in matrix format, where columns are coeff for each task
            Wmat = np.reshape(Wvec, (self.dimension, self.ntasks), order='F')
            # Omega step:
            Omega_old = Omega
#
#            # Learn relationship between tasks (inverse covariance matrix)
            Omega = self.__omega_step(np.cov(Wmat, rowvar=False),
                                      self.lambda_2, self.admm_rho)
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

    def create_permutation_matrix(self, k, v):
        Imat = np.eye(k*v)
        permV = np.zeros((k*v+1, 1),dtype=int)
        cur = 1
        for i in range(1,v+1):
            for j in range(1,k+1):
                permV[cur,0] = i+((j-1)*v)
                cur = cur+1
        permV = permV[1:]-1
        p = Imat[:, permV].T
        return np.squeeze(p)
    
    def create_sparse_AC(self, A, b):
        nmodels = A[0].shape[1]
        ntasks = len(A)
        Acap = np.zeros((nmodels*ntasks, 1))
        C = np.zeros((nmodels*ntasks, 1))

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

    def __omega_step(self, S, lambda_reg, rho):
        '''
            ADMM for estimation of the precision matrix of a Gauss-Markov Random Field.
            
             Input:
                 S: sample covariance matrix
                lambda_reg: regularization parameter (l1-norm)
                rho: dual regularization parameter (default value = 1)
            Output:
                omega: estimated precision matrix
        '''

        # global constants and defaults
        max_iters = 10
        abstol = 1e-5
        reltol = 1e-5
        alpha = 1.4

        # get the number of dimensions
        ntasks = S.shape[0]

        # initiate primal and dual variables
        Z = np.zeros((ntasks, ntasks))
        U = np.zeros((ntasks, ntasks))

        print('[Iters]   Primal Res.  Dual Res.')
        print('------------------------------------')

        for k in range(0, max_iters):

            # x-update
            # numpy returns eigc_val,eig_vec as opposed to matlab's eig
            eig_val, eig_vec = np.linalg.eigh(rho*(Z-U)-S)

            # check eigenvalues
            if isinstance(eig_val[0], complex):
                print("Warning: eigenvalues are complex. Check covariance matrix.")

            # eig_val is already an array (no need to get diag)
            xi = (eig_val + np.sqrt(eig_val**2 + 4*rho)) / (2*rho)
            X = np.dot(np.dot(eig_vec, np.diag(xi, 0)), eig_vec.T)

            # z-update with relaxation
            Zold = Z.copy()
            X_hat = alpha*X + (1-alpha)*Zold
            Z = shrinkage(X_hat+U, lambda_reg/rho)

            # dual variable update
            U = U + (X_hat-Z)

            # diagnostics, reporting, termination checks
            r_norm = np.linalg.norm(X-Z, 'fro')
            s_norm = np.linalg.norm(-rho*(Z-Zold), 'fro')

            eps_pri = np.sqrt(ntasks**2)*abstol + reltol*max(np.linalg.norm(X,'fro'), np.linalg.norm(Z,'fro'))
            eps_dual = np.sqrt(ntasks**2)*abstol + reltol*np.linalg.norm(rho*U,'fro')

            # keep track of the residuals (primal and dual)
            print('   [%d]    %f     %f ' % (k, r_norm, s_norm))
            if r_norm < eps_pri and s_norm < eps_dual:
                break

        return Z

#
#
#    def __objective_function(S, X, rho):
#        obj = 0.5*trace(S*X) - log(det(X)) + rho*norm(X(:), 1)
#        return obj
#

