#-*- coding: utf-8 -*-
import sys
import numpy as np


class FoldIterator:

    def __init__(self, X, y, fold=5):
        assert len(X) == len(y)
        self.X = np.array(np.split(X, fold))
        self.y = np.array(np.split(y, fold))
        self.cur = 0
        self.fold = fold

    def __iter__(self):
        return self

    def __get_fold(self, fold):
        return \
            np.concatenate(self.X[np.arange(self.fold) != fold]), \
            np.concatenate(self.y[np.arange(self.fold) != fold]), \
            self.X[fold], self.y[fold]

    def __next__(self):
        if self.cur < self.fold:
            cur_fold = self.cur
            X, y, X_val, y_val = self.__get_fold(cur_fold)

            self.cur += 1
            return X, y, X_val, y_val, cur_fold + 1
        else:
            raise StopIteration()


sys.modules[__name__] = FoldIterator
