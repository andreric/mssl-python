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

#sys.path.append('..')
#from design import Method


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
        super().__init__('MSSLClassifier', 'MTL')

        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.max_iters = 3000
        self.tol = 1e-5  # minimum tolerance: eps * 100

        self.admm_rho = 1  # ADMM parameter
        self.eps_theta = 1e-3   # stopping criteria parameters
        self.eps_w = 1e-3    # stopping criteria parameters

        self.nb_tasks = -1
        self.dimension = -1
        self.W = None
        self.Omega = None
        self.output_directory = ''

    def fit(self, x, y):

        print('\n[MSSL]')

        # get number of tasks
        self.nb_tasks = len(x)
        self.dimension = x[0].shape[1]

        # create permutation matrix
        P = __create_permutation_matrix(self.dimension, self.ntasks)

        # create (sparse) matrices x and y that are the concatenation
        # of data from all tasks. So, in this way we convert multiple
        # tasks problem to a single (bigger) problem used in the
        # closed-form solution. For gradient-based algorithms like Fista,
        # L-BFGS and others, this is not needed (although can also be used)
        xmat, ymat = __create_sparse_AC(xtrain, ytrain)

        # Learning parameters
        Wc = numpy.linalg.solve(xmat, ymat)  #  warm start
        Omega = np.eye(ntasks)

        # Limited memory BFGS
#        if self.alg_w == 'lbfgs':,
#            opt_fminunc = struct('factr', 1e4, 'pgtol', 1e-5, 'm', 10,'maxIts',10,'printEvery',1e6);
#            if ntasks*dimension > 10000:
#                opt_fminunc.m  = 50

        for it in range(self.max_iters):
            print('%d ' % it)

            # Minimization step
            W_old = Wvec.copy()

            # For classification problems
            Wvec = fminunc(@(w)cost_function_logloss(w, xmat, ymat, self.Omega, P), Wvec, opt)

            # put it in matrix format, where columns are coeff for each task
            Wmat = reshape(Wvec, self.dimension, self.ntasks)

            # Omega step:
            Omega_old = Omega

            # Learn relationship between tasks (inverse covariance matrix)
            Omega = __covsel_admm(cov(Wmat), self.rho2, self.rho)

            # checking convergence of Omega and W
            diff_Omega = norm(Omega-Omega_old, 'fro')
            diff_W = norm(Wvec-W_old)

            # if difference between two consecutive iterations are very small,
            # stop training
            if (diff_Omega < eps_theta) and (diff_W < eps_w):
                break

    def __logloss(w_vec, x_vec, y_vec, theta, perm_matrix):

        ntasks = theta.shape[1]
        ndimension = len(w_vec)/ntasks

        # cost function
        a = np.dot(x_vec, w_vec)
        a = np.maximum(np.minimum(a, 50), -50)
        f = -(y_vec*(a-log(1+exp(a))) + (1-y_vec)*(-log(1+exp(a)))).sum()

        r = reshape(w_vec, ndimension, ntasks)
        f += 0.5*np.trace(np.dot(np.dot(r, theta), r.T))

        # gradient of the cost function
        # logistic regression term
        sig_s = scipy.special.expit(np.dot(x_vec,w_vec))
        g = np.dot(x_vec.T, (sig_s-y_vec))
        matm = np.kron(theta, np.eye(ndimension))

        # regularization term
        g += np.dot(np.dot(np.dot(perm_matrix, matm), perm_matrix.T), w_vec))

        return f, g

    def __create_permutation_matrix(k, v):

        # k: problem dimension
        # v: number of tasks
        sparse = False
        
        if sparse:
            pass
#            I = speye(k*v);
#            permV = spalloc(k*v,1,3*k*v);
        else
            I = np.eye(k*v)
            permV = np.zeros((k*v, 1))

        cur = 1
        for i in range(v):
            for j in range(k):
                permV[cur,0] = i+((j-1)*v)
                cur = cur+1
        p = I[:, permV]
        p = p.T
        return p

    def __create_sparse_AC(A, b):

        sparse = False
        nmodels = A[0].shape[1]
        ntasks = len(A)

        #numModels = size(A,1); #removing the base model
        #v = size(nonNanIndices,1);
        #origSizeObs = size(squeeze(B(:,:,:,1)));
#        if sparse:
#            pass
#            Acap = spalloc(nmodels*ntasks,nmodels*ntasks,3*nmodels*ntasks);
#        else
        Acap = np.zeros((nmodels*ntasks,))

        C = np.zeros((nmodels*ntasks,))

        beginId = 1
        endId = nmodels
        for task in range(ntasks:
            
#            if(sparse)
#                splI = spalloc(ntasks,ntasks,3*ntasks);
#            else
            splI = np.zeros((ntasks, ntasks))

            splI[task][task] = 1 #set this node to 1 so it gets picked up
            
            #do kronecker product and add to Acap
            
            newA = np.dot(A[task].T, A[task])

            Acap = Acap + np.kron(splI, newA)

            C[beginId:endId] = np.dot(A[task].T, b[task])

            beginId = endId+1
            endId = endId+nmodels
        return Acap, C

    def __covsel_admm():
function [Z,f] = covsel_admm( S, lambda, rho )
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

    #Global constants and defaults

    MAX_ITER = 500;
    ABSTOL   = 1e-5;
    RELTOL   = 1e-5;
    ALFA     = 1.4;
    #Data preprocessing

    n = size(S,1);
    #ADMM solver

    #X = zeros(n);
    Z = zeros(n);
    U = zeros(n);

    #if ~QUIET
    #    fprintf('#3s\t#10s\t#10s\t%10s\t%10s\t%10s\n', 'iter', ...
    #      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
    #end

    #dimension = size(D,1);

    for k = 1:MAX_ITER

        # x-update
        #[Q,L] = eig(rho*(Z - U) - (S/2));
        [Q,L] = eig(rho*(Z - U) - S);
        es = diag(L);
        #xi = (es + sqrt(es.^2 + 2*rho*dimension))./(2*rho);
        xi = (es + sqrt(es.^2 + 4*rho))./(2*rho);
        X = Q*diag(xi)*Q';

        # z-update with relaxation
        Zold = Z;
        X_hat = ALFA*X + (1 - ALFA)*Zold;
        Z = shrinkage(X_hat + U, lambda/rho);

        U = U + (X_hat - Z);

        # diagnostics, reporting, termination checks

        #objval(k)  = objective_function(S, X, Z, dimension, lambda);

        r_norm  = norm(X - Z, 'fro');
        s_norm  = norm(-rho*(Z - Zold),'fro');

        eps_pri = sqrt(n*n)*ABSTOL + RELTOL*max(norm(X,'fro'), norm(Z,'fro'));
        eps_dual= sqrt(n*n)*ABSTOL + RELTOL*norm(rho*U,'fro');

        if (r_norm < eps_pri) and (s_norm < eps_dual):
             break

        f = objective_function( S, X, lambda )


    def __objective_function(S, X, rho):
        obj = 0.5*trace(S*X) - log(det(X)) + rho*norm(X(:), 1)
        return obj

    def __shrinkage(a, kappa):
        return np.maximum(0, a-kappa) - np.maximum(0, -a-kappa)
