#-*- coding: utf-8 -*-
import numpy as np

from loguru import logger

import accuracy
import fold_iteration as fi
import neural_network as nn
from enums import Activations, Costs

layers = [
    { "input": 784, "output": 512, "act": Activations.RELU, },
    { "input": 512, "output": 256, "act": Activations.RELU, },
    { "input": 256, "output": 128, "act": Activations.RELU, },
    { "input": 128, "output": 10, "act": Activations.SOFTMAX, },
]


dataset = np.genfromtxt('./dataset/train.csv', delimiter=',', skip_header=1, dtype=int)
X = dataset[:,1:]
truth = dataset[:,:1]
# data 60000x784

print(len(truth))
print(truth[0])
print(type(truth[0]))

# X = np.asarray([np.frombuffer(t, dtype=np.uint8) for t in reader.data])
y = np.zeros((len(truth), 10))
y[np.arange(len(truth)), truth] = 1
_iter = 5
learing_rate = 0.0002

TR_TP = 0
TR_SZ = 0
TR_ACC = 0.0
VAL_TP = 0
VAL_SZ = 0
VAL_ACC = 0.0

for X_train, y_train, X_valid, y_valid, fold in fi(X, y):
    logger.info('fold: {} - started', fold)
    model = nn(layers, Costs.CROSS_ENTROPY)
    model.train(X_train.T, y_train.T,
                X_valid.T, y_valid.T,
                _iter, learing_rate,
                batch_size=0,
                verbose=False)

    tr_tp, tr_sz, tr_acc = accuracy(model.predict(X_train.T), y_train.T)
    TR_TP += tr_tp
    TR_SZ += tr_sz
    TR_ACC += tr_acc
    val_tp, val_sz, val_acc = accuracy(model.predict(X_valid.T), y_valid.T)
    VAL_TP += val_tp
    VAL_SZ += val_sz
    VAL_ACC += val_acc
    logger.info('fold: {} - finished', fold)

logger.info("training micro average = {}/{} = {:.2f}%", TR_TP, TR_SZ, TR_TP * 100.0 / TR_SZ)
logger.info("training macro average = {:.2f}%", TR_ACC / 5.0)
logger.info("valication micro average = {}/{} = {:.2f}%", VAL_TP, VAL_SZ, VAL_TP * 100.0 / VAL_SZ)
logger.info("validation macro average = {:.2f}%", VAL_ACC / 5.0)
# model = nn(layers, Costs.CROSS_ENTROPY)
# model.train(X.T, y.T, _iter, learing_rate, verbose=True)
