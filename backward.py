#-*- coding: utf-8 -*-
import sys
import numpy as np

from loguru import logger

from enums import Activations, Costs


'''
def __sigmoid(dA, A, Z):
    sig = A
    return dA * sig * (1 - sig)
'''


def __relu(dA, A, Z):
    dZ = np.array(dA, copy=True)
    # dZ = np.ones(dA.shape)
    dZ[Z <= 0] = 0
    return dZ


def __softmax(dA, A, Z):
    dZ = np.zeros(dA.shape).T
    _, m = dA.shape
    for i in range(m):
        v = dA[:, i].reshape((-1, 1))
        jac = np.diagflat(v) - np.dot(v, v.T)
        dZ[i] = jac.sum(axis=0)

    return dZ.T


act_map = {
    # Activations.SIGMOID: __sigmoid,
    Activations.RELU: __relu,
    Activations.SOFTMAX: __softmax,
}


def backward(act, dA, A, Z):
    func = act_map[act]
    dZ = func(dA, A, Z)
    return dZ


sys.modules[__name__] = backward
