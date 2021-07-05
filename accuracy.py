#-*- coding: utf-8 -*-
import sys
import numpy as np


def accuracy(y_hat, y, asstring=False):
    pred = y_hat.argmax(axis=0)
    gt = y.argmax(axis=0)
    tp = np.sum(pred == gt)
    sz = len(gt)

    if asstring:
        return '{}/{}={:.2f}%'.format(tp, sz, tp * 100.0 / sz)
    return tp, sz, tp * 100.0 / sz


sys.modules[__name__] = accuracy
