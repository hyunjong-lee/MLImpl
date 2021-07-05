#-*- coding: utf-8 -*-
from enum import Enum


class Activations(Enum):
    SIGMOID = 1
    RELU = 2
    SOFTMAX = 3


class Costs(Enum):
    CROSS_ENTROPY = 1
    SOFTMAX = 2


class Optimizer(Enum):
    SGD = 1
    Momentum = 2
    Nestrov = 3
    Adam = 4
