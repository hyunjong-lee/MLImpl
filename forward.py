#-*- coding: utf-8 -*-
import sys
import numpy as np

from loguru import logger

from enums import Activations, Costs


def __sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def __relu(Z):
    return np.maximum(Z, 0)


def __softmax(Z):
    exp = np.exp(Z)
    denom = np.sum(exp, axis=0, keepdims=True)
    return exp / denom


act_map = {
    Activations.SIGMOID: __sigmoid,
    Activations.RELU: __relu,
    Activations.SOFTMAX: __softmax,
}


def forward(act, A, W, b):
    func = act_map[act]
    Z = np.dot(W, A) + b
    return Z, func(Z)


sys.modules[__name__] = forward
