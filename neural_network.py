#-*- coding: utf-8 -*-
import sys
import numpy as np

from loguru import logger

import accuracy
import cost, cost_diff
import forward, backward
from enums import Activations, Costs, Optimizer


class NeuralNetwork:
    def __init__(self, arch, cost, seed=99):
        self.arch = arch
        self.cost = cost
        self.num_of_layer = len(self.arch)
        self.seed = seed
        self.__init_params()
        self.__init_mem()

    def __init_params(self):
        self.params = {}
        np.random.seed(self.seed)

        for idx, layer in enumerate(self.arch):
            ldx = idx + 1

            in_sz = layer['input']
            out_sz = layer['output']

            self.set_param(ldx, in_sz, out_sz)

    def __init_mem(self):
        self.memory = {}

    def set_param(self, ldx, in_sz, out_sz):
        _he = np.sqrt(2 / in_sz)

        self.params['W%d' % ldx] = np.random \
            .randn(out_sz, in_sz) * _he
        self.params['b%d' % ldx] = np.random \
            .randn(out_sz, 1) * _he

    def get_param(self, _type, ldx):
        param_id = '%s%d' % (_type, ldx)
        return self.params[param_id]

    def predict(self, X, verbose=False):
        A = X
        for idx, layer in enumerate(self.arch):
            ldx = idx + 1
            act = layer['act']
            W = self.get_param('W', ldx)
            b = self.get_param('b', ldx)
            Z, A_next = forward(act, A, W, b)
            A = A_next

        return A

    # TODO
    def save(self, out_path, verbose=False):
        raise NotImplemented

    # TODO
    def load(self, in_path, verbose=False):
        raise NotImplemented

    def train(self, X, y, X_val, y_val, iter, lr, \
              batch_size=0, optimizer=Optimizer.SGD, verbose=False):
        _, m = y.shape

        if verbose:
            logger.debug('verbose option is turned on')
            logger.debug('X.shape: {}', X.shape)
            logger.debug('y.shape: {}', y.shape)
            logger.debug('X_valid.shape: {}', X_val.shape)
            logger.debug('y_valid.shape: {}', y_val.shape)
            logger.debug('# training examples: {}', m)
            logger.debug('iteration: {}', iter)
            logger.debug('learning rate: {}', lr)


        ## TODO apply learning rate using lr
        for it in range(iter):
            if verbose:
                logger.debug('start iteration: {}', it)

            perm = np.random.permutation(m)
            X_T = X.T[perm]
            y_T = y.T[perm]

            sdx = 0
            while True:
                if sdx >= m:
                    break

                if batch_size > 0:
                    X_sam = X_T[sdx:sdx + batch_size].T
                    y_sam = y_T[sdx:sdx + batch_size].T
                    sdx += batch_size
                else:
                    X_sam = X
                    y_sam = y
                    sdx += m

                A = X_sam
                self.memory['A0'] = A
                for idx, layer in enumerate(self.arch):
                    ldx = idx + 1
                    act = layer['act']
                    W = self.get_param('W', ldx)
                    b = self.get_param('b', ldx)
                    Z, A_next = forward(act, A, W, b)
                    A = A_next

                    self.memory['Z%d' % ldx] = Z
                    self.memory['A%d' % ldx] = A

                if verbose:
                    logger.info('iter: {}, cost: {:.2f}, train acc: {}',
                                it,
                                cost(self.cost, y_sam, A),
                                accuracy(A, y_sam, asstring=True))

                # TODO - 아래 수식 정상화 시키기
                # dA = cost_diff(self.cost, y, A)
                dA = []

                for idx, layer in reversed(list(enumerate(self.arch))):
                    ldx = idx + 1

                    # get dZ
                    act = layer['act']
                    if ldx == len(self.arch) and act == Activations.SOFTMAX:
                        dZ = A - y_sam
                    else:
                        Z = self.memory['Z%d' % ldx]
                        A = self.memory['A%d' % ldx]
                        dZ = backward(act, dA, A, Z)

                    A_prev = self.memory['A%d' % idx]
                    W = self.get_param('W', ldx)
                    b = self.get_param('b', ldx)

                    # dA for next iter
                    dA = np.matmul(W.T, dZ)

                    # update current W, b
                    dW = np.matmul(dZ, A_prev.T) / m
                    W = W - lr * dW
                    db = dZ.sum(axis=1, keepdims=True) / m
                    b = b - lr * db
                    # b = b - lr * dW.T.sum(axis=0).reshape((W.shape[0], 1))

                    self.params['W%d' % ldx] = W
                    self.params['b%d' % ldx] = b

            logger.info('iter: {}, cost: {:.2f}, train acc: {}, valid acc: {}',
                it,
                cost(self.cost, y, self.predict(X)),
                accuracy(self.predict(X), y, asstring=True),
                accuracy(self.predict(X_val), y_val, asstring=True))

            if verbose:
                logger.debug('iteration finished: {}', it)


sys.modules[__name__] = NeuralNetwork
