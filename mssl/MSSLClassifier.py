#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:01:39 2018

@author: goncalves1
"""
import sys
import os
import pickle
import numpy as np
import scipy.special
import scipy.optimize
from .MSSL import MSSL

sys.path.append('..')


def weighted_logloss(w, x, y, Omega, lambda_reg, weights):
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
        f3 = np.multiply(f1+f2, weights[t])
        cost += -f3.mean()

    # gradient of regularization term
    cost += (0.5*lambda_reg) * np.trace(np.dot(np.dot(wmat, Omega), wmat.T))
    return cost


def weighted_logloss_der(w, x, y, Omega, lambda_reg, weights):
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
        xweig = np.dot(x[t].T, np.diag(weights[t]))
        grad[:, t] = np.dot(xweig, (sig_s-y[t]))/x[t].shape[0]
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


class MSSLClassifier(MSSL):
    """
    Implement the MSSL classifier.
    """
    def __init__(self, lambda_1=0.1, lambda_2=0,
                 fit_intercept=True, normalize_data=False):
        """ Initialize object with the informed hyper-parameter values. """
        super().__init__(lambda_1, lambda_2, fit_intercept, normalize_data)

    @MSSL._check_inputs  # decorator to chek inputs
    def fit(self, x, y, sample_weight=None):

        self.ntasks = len(x)  # get number of tasks
        self.ndimensions = x[0].shape[1]  # dimension of the data
        if self.fit_intercept:
            self.ndimensions += 1  # if consider intercept, add another feat +1

        x, y, offsets = self.preprocess_data(x, y)
        self.offsets = offsets

        if sample_weight is None:
            sample_weight = np.ones(x.shape[0])

        W, Omega = self.__train(x, y, sample_weight,
                                weighted_logloss,
                                weighted_logloss_der)

        self.W = W.copy()
        self.Omega = Omega.copy()
        fname = os.path.join(self.output_directory, '%s.mdl' % self.__str__())
        with open(fname, 'wb') as fh:
            pickle.dump([self.W, self.Omega], fh)

    def predict(self, x, **kwargs):
        for t in range(self.ntasks):
            x[t] = x[t].astype(np.float64)
            x[t] = (x[t]-self.offsets['x_offset'][t])
            if self.normalize_data:
                x[t] = x[t]/self.offsets['x_scale'][t]
            if self.fit_intercept:
                x[t] = np.hstack((x[t], np.ones((x[t].shape[0], 1))))

        yhat = [None]*len(x)
        for t in range(len(x)):
            yhat[t] = scipy.special.expit(np.dot(x[t], self.W[:, t]))
            yhat[t] = yhat[t] #np.around().astype(np.int32)
        return yhat
