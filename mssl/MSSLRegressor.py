#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:41:50 2018

@author: goncalves1
"""
import os
import pickle
import numpy as np
from .MSSL import MSSL


class MSSLRegressor(MSSL):
    """
    Implement the MSSL Regressor.
    """
    def __init__(self, lambda_1=0.1, lambda_2=0,
                 fit_intercept=True, normalize_data=False):
        """ Initialize object with the informed hyper-parameter values. """
        super().__init__(lambda_1, lambda_2, fit_intercept, normalize_data)

    def fit(self, x, y, sample_weight=None):

        self._check_inputs(x, y, sample_weight)

        self.ntasks = len(x)  # get number of tasks
        self.ndimensions = x[0].shape[1]  # dimension of the data
        if self.fit_intercept:
            self.ndimensions += 1  # if consider intercept, add another feat +1

        x, y, w, self.offsets = self._preprocess_data(x, y, sample_weight)

        W, Omega = self._train(x, y, w,
                                weighted_squaredloss,
                                weighted_squaredloss_der)
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
            yhat[t] = np.dot(x[t], self.W[:, t])
            yhat[t] = np.maximum(0, yhat[t])
        return yhat


def weighted_squaredloss(w, x, y, Omega, lambda_reg, weights):
    ''' MSSL with logloss function '''
    ntasks = Omega.shape[1]
    ndimension = int(len(w)/ntasks)
    wmat = np.reshape(w, (ndimension, ntasks), order='F')

    # cost function for each task
    cost = 0
    for t in range(ntasks):
        h_t_x = np.dot(x[t], wmat[:, t])
        cost += np.multiply((h_t_x - y[t])**2, weights[t]).mean()

    # cost for the regularization term
    cost += (0.5*lambda_reg) * np.trace(np.dot(np.dot(wmat, Omega), wmat.T))
    return cost


def weighted_squaredloss_der(w, x, y, Omega, lambda_reg, weights):
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
        U_t = np.diag(weights[t]).copy()
        g1 = np.dot(x[t].T, U_t)
        g2 = np.dot(x[t], wmat[:, t]) - y[t]
        grad[:, t] = (2.0/x[t].shape[0])*np.dot(g1, g2)

    # gradient of regularization term
    grad += lambda_reg * np.dot(wmat, Omega)
    grad = np.reshape(grad, (ndimension*ntasks, ), order='F')
    return grad