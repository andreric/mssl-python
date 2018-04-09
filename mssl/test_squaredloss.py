#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 19:50:28 2018

@author: andre
"""

import pickle
import numpy as np
import scipy.optimize as spo
#import scipy.special


def squaredloss(w, x, y, Omega, lambda_reg):
    '''MSSL with squared loss function '''
    ntasks = Omega.shape[1]
    ndimension = int(len(w)/ntasks)
    wmat = np.reshape(w, (ndimension, ntasks), order='F')

    # make sure data is in the correct format
    for t in range(ntasks):
        if len(y[t].shape) > 1:
            y[t] = np.squeeze(y[t])

    # cost function for each task
    cost = 0
    for t in range(ntasks):
        h_t_x = np.dot(x[t], wmat[:, t])
        cost += ((h_t_x - y[t])**2).mean()
    # cost function regularization term
    cost += (0.5*lambda_reg) * np.trace(np.dot(np.dot(wmat, Omega), wmat.T))
    return cost

def squaredloss_der(w, x, y, Omega, lambda_reg):
    ''' Gradient of the MSSL with squared loss function '''

    ntasks = Omega.shape[1]
    ndimension = int(len(w)/ntasks)
    wmat = np.reshape(w, (ndimension, ntasks), order='F')

    # make sure data is in correct format
    for t in range(ntasks):
        if len(y[t].shape) > 1:
            y[t] = np.squeeze(y[t])

    # gradient of squared loss term
    grad = np.zeros(wmat.shape)
    for t in range(ntasks):
        g1 = np.dot(np.dot(x[t].T, x[t]), wmat[:, t])
        g2 = np.dot(x[t].T, y[t])
        grad[:, t] = (2.0/x[t].shape[0])*(g1-g2)
    # gradient of regularization term
    grad += lambda_reg * np.dot(wmat, Omega)
    grad = np.reshape(grad, (ndimension*ntasks, ), order='F')
    return grad


if __name__ == '__main__':
    
    # Classification
    print('Loading data ...')
    with open('../datasets/toy_10tasks_clf.pkl', 'rb') as fh:
        x, y, dimension, ntasks = pickle.load(fh)

    ntasks = len(x)  # get number of tasks
    dimension = x[0].shape[1]  # get problem dimension

    Omega = np.eye(ntasks)
    lambda_reg = 0.5
    w = -0.05 + 0.1*np.random.rand(dimension*ntasks, )
    eps = [np.sqrt(np.finfo(float).eps)]*ntasks*dimension
    r = spo.check_grad(squaredloss, squaredloss_der, w, x, y, Omega, lambda_reg)
    print(r)