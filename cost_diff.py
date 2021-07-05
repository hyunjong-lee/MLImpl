#-*- coding: utf-8 -*-
import sys
import numpy as np

from enums import Costs


def __cross_entropy(y, y_hat):
    return -y / y_hat


cost_map = {
    Costs.CROSS_ENTROPY: __cross_entropy,
}


def cost(c, y, y_hat):
    func = cost_map[c]
    return func(y, y_hat)


sys.modules[__name__] = cost
