#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:54:33 2018

@author: goncalves1
"""
import sys
import numpy as np
import scipy.special
import scipy.optimize
import scipy.sparse
import scipy.linalg
sys.path.append('..')
from design import Method


class MSSLRegressor(Method):
    """
    Implement the MSSL classifier.

    Attributes:
        lambda_1 (float): Omega penalization hyper-parameter
        lambda_2 (float): W l1-penalization hyper-parameter
    """
    def __init__(self, lambda_1=0.1, lambda_2=0):
        """ Initialize object with the informed hyper-parameter values.
        Args:
            lambda_1 (float): Omega penalization hyper-parameter
            lambda_2 (float): W l1-penalization hyper-parameter
        """
        # set method's name and paradigm
        super().__init__('MSSLRegressor', 'MTL')

        self.wstep_alg = 'closed-form'

        self.lambda_1 = lambda_1  # omega sparsity
        self.lambda_2 = lambda_2  # w sparsity
        self.max_iters = 100
        self.tol = 1e-5  # minimum tolerance: eps * 100

        self.sparse = False
        self.normalize_data = True
        self.admm_rho = 1  # ADMM parameter
        self.eps_theta = 1e-3  # stopping criteria parameters
        self.eps_w = 1e-3  # stopping criteria parameters

        self.ntasks = -1
        self.dimension = -1
        self.W = None
        self.Omega = None
        self.output_directory = ''

    def fit(self, x, y):

        # get number of tasks
        self.ntasks = len(x)
        self.dimension = x[0].shape[1]

        x, y, offsets = self.__preprocess_data(x, y)
        self.offsets = offsets

        W, Omega = self.__mssl_train(x, y)
        self.W = W.copy()
        self.Omega = Omega.copy()

    def predict(self, x):
        for t in range(self.ntasks):
            x[t] = x[t].as_matrix().astype(np.float64)
            x[t] = (x[t]-self.offsets['x_offset'][t])
            if self.normalize_data:
                x[t] = x[t]/self.offsets['x_scale'][t]

        yhat = [None]*len(x)
        for t in range(len(x)):
            yhat[t] = np.dot(x[t], self.W[:, t]) + self.offsets['y_offset'][t]
            yhat[t] = np.maximum(0, yhat[t])
        return yhat

    def __preprocess_data(self, x, y):

        # make sure y is in correct shape
        for t in range(self.ntasks):
            x[t] = x[t].as_matrix().astype(np.float64)
            y[t] = y[t].as_matrix().astype(np.float64).ravel()
            if len(y[t].shape) < 2:
                y[t] = y[t][:, np.newaxis]

        offsets = {'x_offset': list(),
                   'x_scale': list(),
                   'y_offset': list()}
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

            offsets['y_offset'].append(y[t].mean(axis=0))
            y[t] = y[t] - offsets['y_offset'][t]
        return x, y, offsets

    def __mssl_train(self, x, y):

        # create permutation matrix
        P = self.create_permutation_matrix(self.dimension, self.ntasks)

        # create (sparse) matrices x and y that are the concatenation
        # of data from all tasks. So, in this way we convert multiple
        # tasks problem to a single (bigger) problem used in the
        # closed-form solution. For gradient-based algorithms like Fista,
        # L-BFGS and others, this is not needed (although can also be used)
        xmat, ymat = self.create_sparse_AC(x, y)

        # initialize learning parameters
#        Wvec =  scipy.linalg.lstsq(xmat, ymat)   #  regression warm start
        Wvec = -0.05 + 0.05*np.random.rand(self.ntasks*self.dimension, 1)
        Omega = np.eye(self.ntasks)

        # scipy opt parameters
        for it in range(self.max_iters):

            # Minimization step
            W_old = Wvec.copy()

            if self.wstep_alg == 'closed-form':
                factor = self.lambda_2
                L = np.kron(np.eye(self.dimension), Omega)
                Xls = xmat + np.dot(np.dot(factor * P, L), P.T)
#                if Xls.shape[0] == Xls.shape[1]:  # squared Xls
#                    Wvec = np.linalg.solve(Xls, ymat)
#                else:
                Wvec, _, _, _ = scipy.linalg.lstsq(Xls, ymat)

            elif self.wstep_alg == 'gradient-based':
                opts = {'maxiter': 10, 'disp': True}
                additional = (xmat, ymat, Omega, P)
                res = scipy.optimize.minimize(squared_loss, x0=Wvec,
                                              args=additional,
                                              jac=True, method='BFGS',
                                              options=opts)
                Wvec = res.x.copy()
            else:
                raise NotImplementedError("W-step %s not found" %
                                          self.wstep_alg)

            # put it in matrix format, where columns are coeff for each task
            Wmat = np.reshape(Wvec, (self.dimension, self.ntasks), order='F')
            # Omega step:
            Omega_old = Omega

            # Learn relationship between tasks (inverse covariance matrix)
            Omega = self.__omega_step(np.cov(Wmat, rowvar=False),
                                      self.lambda_1, self.admm_rho)

            # checking convergence of Omega and W
            diff_Omega = np.linalg.norm(Omega-Omega_old, 'fro')
            diff_W = np.linalg.norm(Wvec-W_old)

            # if difference between two consecutive iterations are very small,
            # stop training
            if (diff_Omega < self.eps_theta) and (diff_W < self.eps_w):
                break
        return Wmat, Omega

    def set_params(self, params):
        """
        Set hyper-parameters to be used in the execution.
        Args:
            params (dict): dict with hyper-parameter values.
        """
        self.lambda_1 = params['lambda_1']
        self.lambda_2 = params['lambda_2']

    def get_params(self):
        """ Return hyper-parameters used in the execution.
        Return:
            params (dict): dict containing the hyper-params values.
        """
        ret = {'lambda_1': self.lambda_1,
               'lambda_2': self.lambda_2}
        return ret

    def get_params_grid(self):
        """ Yield set of hyper-parameters to be tested out."""
        lambda_1 = np.logspace(-7, 2, 10)
        lambda_2 = np.logspace(0, 4, 10)
        for r0 in lambda_1:
#            yield {'lambda_1': r0, 'lambda_2': 0}
            for r1 in lambda_2:
                yield {'lambda_1': r0,
                       'lambda_2': r1}

    def set_output_directory(self, output_dir):
        """ Set output folder path.
        Args:
            output_dir (str): path to output directory.
        """
        self.output_directory = output_dir
        self.logger.set_path(output_dir)
        self.logger.setup_logger(self.__str__())

    def create_permutation_matrix(self, k, v):
        if self.sparse:
            Imat = scipy.sparse.eye(k*v)
            permV = scipy.sparse.csr_matrix((k*v+1, 1))
        else:
            Imat = np.eye(k*v)
            permV = np.zeros((k*v+1, 1), dtype=int)
        cur = 1
        for i in range(1, v+1):
            for j in range(1, k+1):
                permV[cur, 0] = i+((j-1)*v)
                cur = cur+1
        permV = permV[1:]-1
        p = Imat[:, permV].T
        return np.squeeze(p)

    def create_sparse_AC(self, A, b):
        nmodels = A[0].shape[1]
        ntasks = len(A)

        if self.sparse:
            Acap = scipy.sparse.csr_matrix((nmodels*ntasks, nmodels*ntasks))
        else:
            Acap = np.zeros((nmodels*ntasks, 1))
        C = np.zeros((nmodels*ntasks, 1))

        beginId = 0
        endId = nmodels
        for task in range(ntasks):
            splI = np.zeros((ntasks, ntasks))
            splI[task][task] = 1  # set this node to 1 so it gets picked up
            # do kronecker product and add to Acap
            newA = np.dot(A[task].T, A[task])
            Acap = Acap + np.kron(splI, newA)
            C[beginId:endId] = np.dot(A[task].T, b[task])
            beginId = endId
            endId = endId+nmodels
        return Acap, C

    def __omega_step(self, S, lambda_reg, rho):
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

        # get the number of dimensions
        ntasks = S.shape[0]

        # initiate primal and dual variables
        Z = np.zeros((ntasks, ntasks))
        U = np.zeros((ntasks, ntasks))

#        print('[Iters]   Primal Res.  Dual Res.')
#        print('------------------------------------')

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

            # dual variable update
            U = U + (X_hat-Z)

            # diagnostics, reporting, termination checks
            r_norm = np.linalg.norm(X-Z, 'fro')
            s_norm = np.linalg.norm(-rho*(Z-Zold), 'fro')

            eps_pri = np.sqrt(ntasks**2)*abstol + reltol*max(np.linalg.norm(X,'fro'), np.linalg.norm(Z,'fro'))
            eps_dual = np.sqrt(ntasks**2)*abstol + reltol*np.linalg.norm(rho*U,'fro')

            # keep track of the residuals (primal and dual)
#            print('   [%d]    %f     %f ' % (k, r_norm, s_norm))
            if r_norm < eps_pri and s_norm < eps_dual:
                break

        return Z


def squared_loss(w, x, y, Omega, P):

    ntasks = Omega.shape[1]
    ndimension = int(len(w)/ntasks)

    lambda_reg = 1

    wmat = np.reshape(w, (ndimension, ntasks), order='F')
    # cost function
    f = 0.5*np.dot(np.dot(w.T, x), w) - np.dot(w.T, y)
    f += (0.5*lambda_reg)*np.trace(np.dot(np.dot(wmat, Omega), wmat.T))

    # gradient
    km = np.kron(Omega, np.eye(ndimension))
    g = np.dot(x.T + np.dot(np.dot(lambda_reg*P, km), P.T), w) - y
    return f, g


def shrinkage(a, kappa):
    return np.maximum(0, a-kappa) - np.maximum(0, -a-kappa)

#if strcmp(prob_type,'regression'),
#            
#            switch( opts_ao.alg_w ),
#                
#                case 'fminunc',
#                    Wvec = fminunc( @(w)cost_function_squared_loss(w, x, y, Omega, P), Wvec, opt );
#                
#                case 'fista', 
#                    
#                    [Wvec,~] = acc_proximal_gradient( Wvec, x, y, Omega, P, opts_ao.gamma, 10 );
#
#                case 'lbfgs',
#                    opts.x0 = Wvec;
#                    [Wvec, ~, info] = lbfgsb(@(w)cost_function_squared_loss(w, x, y, Omega, P), ...
#                                          -inf(ntasks*dimension,1), inf(ntasks*dimension,1), opts );
#                
#                otherwise
#                    error('Optimization algorithm for W-step is not valid!');
#                    
#                
#            end
#            
