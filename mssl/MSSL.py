#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:43:00 2018

@author: goncalves1
"""
import numpy as np
import scipy.optimize


class MSSL(object):
    def __init__(self, lambda_1=0.1, lambda_2=0,
                 fit_intercept=True, normalize_data=False):
        """ Initialize object with the informed hyper-parameter values. """

        self.lambda_1 = lambda_1  # trace term
        self.lambda_2 = lambda_2  # omega sparsity
        self.max_iters = 100
        self.tol = 1e-4  # minimum tolerance: eps * 100

        self.admm_rho = 1  # ADMM parameter
        self.eps_theta = 1e-4  # stopping criteria parameters
        self.eps_w = 1e-4  # stopping criteria parameters

        self.fit_intercept = fit_intercept
        self.normalize_data = normalize_data

        self.ntasks = -1
        self.ndimensions = -1
        self.W = None
        self.Omega = None
        self.output_directory = ''
    
    def _check_inputs(self, x, y, w):
        """ Check if x, y, and w have matching dimensions""" 
        assert len(x) == len(y)
        if w is not None:
            assert len(x) == len(w)
        for k in range(len(x)):
            assert x[k].shape[0] == y[k].shape[0]
            if w is not None:
                assert y[k].shape[0] == w[k].shape[0]
            
        
    def _train(self, x, y, weights,
                weighted_costfunction,
                weighted_costfunction_der):

        # initialize learning parameters
        # np.linalg.solve(xmat, ymat)  #  regression warm start
        W = -0.05 + 0.05*np.random.rand(self.ndimensions, self.ntasks)
        Omega = np.eye(self.ntasks)

        # scipy opt parameters
        opts = {'maxiter': 5, 'disp': False}
        for it in range(self.max_iters):
#            print('%d ' % it)

            # Minimization step
            W_old = W.copy()
            Wvec = np.reshape(W, (self.ndimensions*self.ntasks, ), order='F')

#            r = scipy.optimize.check_grad(weighted_logloss,
#                                          weighted_logloss_der,
#                                          Wvec, x, y,
#                                          Omega, self.lambda_1, weights)
#            print(r)

            additional = (x, y, Omega, self.lambda_1, weights)
            res = scipy.optimize.minimize(weighted_costfunction, x0=Wvec,
                                          args=additional,
                                          jac=weighted_costfunction_der,
                                          method='BFGS',
                                          options=opts)

            # put it in matrix format, where columns are coeff for each task
            W = np.reshape(res.x.copy(),
                           (self.ndimensions, self.ntasks), order='F')
            # Omega step:
            Omega_old = Omega

            # Learn relationship between tasks (inverse covariance matrix)
            Omega = self._omega_step(np.cov(W, rowvar=False),
                                      self.lambda_2, self.admm_rho)

            # checking convergence of Omega and W
            diff_Omega = np.linalg.norm(Omega-Omega_old)
            diff_W = np.linalg.norm(W-W_old)

            # if difference between two consecutive iterations are very small,
            # stop training
            if (diff_Omega < self.eps_theta) and (diff_W < self.eps_w):
                break

        return W, Omega

    def _omega_step(self, S, lambda_reg, rho):
        '''
        ADMM for estimation of the precision matrix.

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

        # varying penalty parameter (rho)
        mu = 10
        tau_incr = 2
        tau_decr = 2

        # get the number of dimensions
        ntasks = S.shape[0]

        def shrinkage(a, kappa):
            return np.maximum(0, a-kappa) - np.maximum(0, -a-kappa)

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
                print("Warning: complex eigenvalues. Check covariance matrix.")

            # eig_val is already an array (no need to get diag)
            xi = (eig_val + np.sqrt(eig_val**2 + 4*rho)) / (2*rho)
            X = np.dot(np.dot(eig_vec, np.diag(xi, 0)), eig_vec.T)

            # z-update with relaxation
            Zold = Z.copy()
            X_hat = alpha*X + (1-alpha)*Zold
            Z = shrinkage(X_hat+U, lambda_reg/rho)
#            Z = Z - np.diag(np.diag(Z)) + np.eye(Z.shape[0])
            # dual variable update
            U = U + (X_hat-Z)

            # diagnostics, reporting, termination checks
            r_norm = np.linalg.norm(X-Z, 'fro')
            s_norm = np.linalg.norm(-rho*(Z-Zold), 'fro')

#            if r_norm > mu*s_norm:
#                rho = rho*tau_incr
#            elif s_norm > mu*r_norm:
#                rho = rho/tau_decr

            eps_pri = np.sqrt(ntasks**2)*abstol + reltol*max(np.linalg.norm(X,'fro'), np.linalg.norm(Z,'fro'))
            eps_dual = np.sqrt(ntasks**2)*abstol + reltol*np.linalg.norm(rho*U,'fro')

            # keep track of the residuals (primal and dual)
            print('   [%d]    %f     %f ' % (k, r_norm, s_norm))
            if r_norm < eps_pri and s_norm < eps_dual:
                break

        return Z

    def _preprocess_data(self, x, y, sample_weight):

        if sample_weight is None:
            w = list()
            for xi in x:
                w.append(np.ones(xi.shape[0]))
        
        # make sure y is in correct shape
        offsets = {'x_offset': list(),
                   'x_scale': list()}
        for t in range(self.ntasks):
            offsets['x_offset'].append(x[t].mean(axis=0))
            if self.normalize_data:
                std = x[t].std(axis=0)
                std[std == 0] = 1
                offsets['x_scale'].append(std)
            else:
                std = np.ones((x[t].shape[1],))
                offsets['x_scale'].append(std)
            x[t] = (x[t] - offsets['x_offset'][t]) / offsets['x_scale'][t]
            if self.fit_intercept:
                x[t] = np.hstack((x[t], np.ones((x[t].shape[0], 1))))
        return x, y, w, offsets


    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())
