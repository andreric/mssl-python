#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:42:16 2018

@author: goncalves1
"""
import pickle
import numpy as np
import scipy.optimize as spo
import scipy.special
#from MSSLClassifier import logloss
from MSSLClassifier import MSSLClassifier
from sklearn.metrics import log_loss

def logloss(w, x, y, Omega, P):

#    x, y, Omega, P = args['x'], args['y'], args['Omega'], args['P']

    ntasks = Omega.shape[1]
    ndimension = int(len(w)/ntasks)

    # cost function
    a = np.maximum(np.minimum(np.dot(x, w), 50), -50)
    f1 = np.multiply(y, a-np.log(1+np.exp(a)))
    f2 = np.multiply(1-y, -np.log(1+np.exp(a)))
    f = -(f1 + f2).sum()
#    f = log_loss(y, a)

#    r = np.reshape(w, (ndimension, ntasks), order='F')
#    f += 0.5*np.trace(np.dot(np.dot(r, Omega), r.T))
    return f

def logloss_der(w, x, y, Omega, P):
    ntasks = Omega.shape[1]
    ndimension = int(len(w)/ntasks)

    # gradient of the cost function
    # logistic regression term
#    sig_s = scipy.special.expit(np.dot(x,w))[:, np.newaxis]
    sig_s = sigmoid(np.dot(x, w))[:, np.newaxis]
    g0 = np.dot(x.T, (sig_s-y))
#    matm = np.kron(Omega, np.eye(ndimension))

    # regularization term
#    g1 = np.dot(np.dot(np.dot(P, matm), P.T), w)
    g = g0[:, 0]# + g1
    return g

def sigmoid(a):
    # Logit function for logistic regression
    # x: data point (array or a matrix of floats)
    # w: set of weights (array of float)

    # treating over/underflow problems
    a = np.maximum(np.minimum(a, 50), -50)
    #f = 1./(1+exp(-a));
    f = np.exp(a) / (1+np.exp(a))
    return f


if __name__ == '__main__':
    
    # Classification
    print('Loading data ...')
    with open('../datasets/toy_10tasks_clf.pkl', 'rb') as fh:
        x, y, dimension, ntasks = pickle.load(fh)
    
    # get number of tasks
    ntasks = len(x)
    dimension = x[0].shape[1]
    
    mssl_clf = MSSLClassifier()
    
    # create permutation matrix
    P = mssl_clf.create_permutation_matrix(dimension, ntasks)
    
    # create (sparse) matrices x and y that are the concatenation
    # of data from all tasks. So, in this way we convert multiple
    # tasks problem to a single (bigger) problem used in the
    # closed-form solution. For gradient-based algorithms like Fista,
    # L-BFGS and others, this is not needed (although can also be used)
    xmat, ymat = mssl_clf.create_sparse_AC(x, y)
    
    Omega = np.eye(ntasks)
    
    w = -0.05 + 0.1*np.random.rand(ntasks*dimension, 1)
    eps = [np.sqrt(np.finfo(float).eps)]*ntasks*dimension
    r = spo.check_grad(logloss, logloss_der, w[:, 0], xmat, ymat[:, 0], Omega, P)
    print(r)