#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 08:51:38 2018

@author: goncalves1
"""

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

sys.path.append('..')
from design import Method


def logloss(w, x, y, Omega, lambda_reg):
    ''' MSSL with logloss function '''
    ntasks = Omega.shape[1]
    ndimension = int(len(w)/ntasks)
    wmat = np.reshape(w, (ndimension, ntasks), order='F')

    # make sure the data is in the correct format
    for t in range(ntasks):
        if len(y[t].shape) > 1:
            y[t] = np.squeeze(y[t])

    # cost function for each task
    cost = 0
    for t in range(ntasks):
        h_t_x = sigmoid(np.dot(x[t], wmat[:, t]))
#       h_t_x = scipy.special.expit(np.dot(x[t], wmat[:, t]))
        f1 = np.multiply(y[t], np.log(h_t_x))
        f2 = np.multiply(1-y[t], np.log(1-h_t_x))
        cost += -(f1 + f2).mean()

    # gradient of regularization term
    cost += (0.5*lambda_reg) * np.trace(np.dot(np.dot(wmat, Omega), wmat.T))
    return cost


def logloss_der(w, x, y, Omega, lambda_reg):
    ''' Gradient of the MSSL with logloss function '''

    ntasks = Omega.shape[1]
    ndimension = int(len(w)/ntasks)
    wmat = np.reshape(w, (ndimension, ntasks), order='F')

    # make sure data is in correct format
    for t in range(ntasks):
        if len(y[t].shape) > 1:
            y[t] = np.squeeze(y[t])

    # gradient of logloss term
    grad = np.zeros(wmat.shape)
    for t in range(ntasks):
#        sig_s = scipy.special.expit(np.dot(x[t], wmat[:, t])) #[:, np.newaxis]
        sig_s = sigmoid(np.dot(x[t], wmat[:, t]))
        grad[:, t] = np.dot(x[t].T, (sig_s-y[t]))/x[t].shape[0]
    # gradient of regularization term
    grad += lambda_reg * np.dot(wmat, Omega)
    grad = np.reshape(grad, (ndimension*ntasks, ), order='F')
    return grad


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


class MSSLClassifier(Method):
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
        super().__init__('MSSLClassifier', 'MTL')

        self.lambda_1 = lambda_1  # trace term
        self.lambda_2 = lambda_2  # omega sparsity
        self.max_iters = 100
        self.tol = 1e-4  # minimum tolerance: eps * 100

        self.normalize_data = True
        self.admm_rho = 20000  # ADMM parameter
        self.eps_theta = 1e-3  # stopping criteria parameters
        self.eps_w = 1e-3  # stopping criteria parameters

        self.ntasks = -1
        self.ndimensions = -1
        self.W = None
        self.Omega = None
        self.output_directory = ''

    def fit(self, x, y, intercept=True):

        self.ntasks = len(x)  # get number of tasks
        self.ndimensions = x[0].shape[1]  # dimension of the data
        if intercept:
            self.ndimensions += 1  # if consider intercept, add another feat +1
        self.intercept = intercept

        x, y, offsets = self.__preprocess_data(x, y)
        self.offsets = offsets

        W, Omega = self.__mssl_train(x, y)
        self.W = W.copy()
        self.Omega = Omega.copy()
        fname = os.path.join(self.output_directory, '%s.mdl' % self.__str__())
        with open(fname, 'wb') as fh:
            pickle.dump([self.W, self.Omega], fh)

    def predict(self, x):
        for t in range(self.ntasks):
            x[t] = x[t].astype(np.float64)
            x[t] = (x[t]-self.offsets['x_offset'][t])
            if self.normalize_data:
                x[t] = x[t]/self.offsets['x_scale'][t]
            if self.intercept:
                x[t] = np.hstack((x[t], np.ones((x[t].shape[0], 1))))

        yhat = [None]*len(x)
        for t in range(len(x)):
            yhat[t] = scipy.special.expit(np.dot(x[t], self.W[:, t]))
            yhat[t] = np.around(yhat[t]).astype(np.int32)
        return yhat

    def __preprocess_data(self, x, y):
        # make sure y is in correct shape
        for t in range(self.ntasks):
            x[t] = x[t].astype(np.float64)
            y[t] = y[t].astype(np.float64).ravel()
            if len(y[t].shape) < 2:
                y[t] = y[t][:, np.newaxis]

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
            if self.intercept:
                x[t] = np.hstack((x[t], np.ones((x[t].shape[0], 1))))
        return x, y, offsets

    def __mssl_train(self, x, y):

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

#            r = scipy.optimize.check_grad(logloss,
#                                          logloss_der,
#                                          Wvec, x, y, Omega, self.lambda_1)
#            print(r)

            additional = (x, y, Omega, self.lambda_1)
            res = scipy.optimize.minimize(logloss, x0=Wvec,
                                          args=additional,
                                          jac=logloss_der,
                                          method='BFGS',
                                          options=opts)

            # put it in matrix format, where columns are coeff for each task
            W = np.reshape(res.x.copy(), 
                           (self.ndimensions, self.ntasks), order='F')
            # Omega step:
            Omega_old = Omega

            # Learn relationship between tasks (inverse covariance matrix)
            Omega = self.__omega_step(np.cov(W, rowvar=False),
                                      self.lambda_2, self.admm_rho)

            # checking convergence of Omega and W
            diff_Omega = np.linalg.norm(Omega-Omega_old)
            diff_W = np.linalg.norm(W-W_old)

            # if difference between two consecutive iterations are very small,
            # stop training
            if (diff_Omega < self.eps_theta) and (diff_W < self.eps_w):
                break

        return W, Omega


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
        print('lambda_reg: %f'%lambda_reg)
        print('rho: %f'%rho)
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
#            print('   [%d]    %f     %f ' % (k, r_norm, s_norm))
            if r_norm < eps_pri and s_norm < eps_dual:
                break

        return Z

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
        lambda_1 = np.logspace(-1, 3, 10)
        lambda_2 = np.logspace(-5, 2, 10)
        for r0 in lambda_1:
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


#def logloss(w, x, y, Omega, lambda_reg):
#
#    ntasks = Omega.shape[1]
#    ndimensions = int(len(w)/ntasks)
#    wmat = np.reshape(w, (ndimensions, ntasks), order='F')
#
#    for t in range(ntasks):
#        if len(y[t].shape) > 1:
#            y[t] = np.squeeze(y[t])
#
#    # cost function
#    cost = 0
#    grad = np.zeros(wmat.shape)
#    for t in range(ntasks):
#        h_t_x = sigmoid(np.dot(x[t], wmat[:, t]))
##       h_t_x = scipy.special.expit(np.dot(x[t], wmat[:, t]))
#        f1 = np.multiply(y[t], np.log(h_t_x))
#        f2 = np.multiply(1-y[t], np.log(1-h_t_x))
#        cost += -(f1 + f2).mean()
#        
#        grad[:, t] = np.dot(x[t].T, (h_t_x-y[t]))/x[t].shape[0]
#
#    # cost function regularization
#    cost += (0.5*lambda_reg) * np.trace(np.dot(np.dot(wmat, Omega), wmat.T))
#    # grad regularization
#    grad += lambda_reg * np.dot(wmat, Omega)
#    grad = np.reshape(grad, (ndimensions*ntasks, ), order='F')
#    return cost, grad
