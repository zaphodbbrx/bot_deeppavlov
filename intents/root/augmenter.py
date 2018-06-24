#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 12:24:43 2018

@author: lsm
"""

from overrides import overrides

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.serializable import Serializable

log = get_logger(__name__)

@register('augmenter')
class AugmenterGaussian(Component, Serializable):
    def __init__(self, mean, power, num_noise, **kwargs):
        self.mean = mean
        self.power = power
        self.num_noise = num_noise
        self.mode=kwargs['mode']
        
    def save(self, *args, **kwargs):
        raise NotImplementedError
        
    def load(self, *args, **kwargs):
        pass        
    @overrides
    def __call__(self, xv, y, *args):#, **kwargs):
        return [xv,y]
    def process_event(self, event_name, data):
        pass
    def train_on_batch(self, xv, y, *args, **kwargs):
        fn, ln = self.__add_noise(xv, y, self.mean, self.power, self.num_noise)
        return fn,ln

    def __add_noise(self, feats, labels, mean, power, num_noise):
        fn = feats[0]
        ln = labels[0]
        for i in range(int(num_noise)):
            noise = np.random.normal(mean, power, feats[0].shape)
            noised = feats[0]*noise
            fn = np.vstack([fn,noised])
            ln = np.vstack([ln,labels[0]])
        return fn,ln
