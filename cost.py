#-*- coding: utf-8 -*-
import sys
import numpy as np

from enums import Costs


def __cross_entropy(y, y_hat):
    lyh = np.log(y_hat + 1e-9)
    mul = -np.multiply(y, lyh)
    tot = mul.sum(axis=0, dtype='float')
    return np.average(tot)


cost_map = {
    Costs.CROSS_ENTROPY: __cross_entropy,
}


def cost(c, y, y_hat):
    func = cost_map[c]
    return func(y, y_hat)


sys.modules[__name__] = cost
