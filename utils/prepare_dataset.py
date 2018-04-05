#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:59:03 2018

@author: andre
"""

import scipy.io
import scipy.special
import numpy as np
import os
import pickle

path = '/home/andre/Downloads/andreric-mssl-code-31a05d9a4517/src/datasets/'

mat = scipy.io.loadmat(os.path.join(path, 'toy_10tasks.mat'))

print(mat.keys())
x = list()
y = list()
ybin = list()
for i in range(10):
    x.append(mat['x'][0][i])
    y.append(mat['y'][0][i])
    yi = np.maximum(0, np.sign(mat['y'][0][i]))
    ybin.append(yi.astype(int))

dimension = mat['d'][0][0]
ntasks = mat['ntasks'][0][0]

with open('/home/andre/Documents/repo/mssl-python/datasets/toy_10tasks_reg.pkl', 'wb') as fh:
    pickle.dump([x, y, dimension, ntasks], fh)

y = ybin
with open('/home/andre/Documents/repo/mssl-python/datasets/toy_10tasks_clf.pkl', 'wb') as fh:
    pickle.dump([x, y, dimension, ntasks], fh)