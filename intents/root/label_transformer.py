#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:40:06 2018

@author: lsm
"""
import sys
from overrides import overrides

import numpy as np
import pickle

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.serializable import Serializable
from keras.utils import to_categorical


log = get_logger(__name__)

@register('label_transformer')
class LabelTransformer(Estimator, Serializable):
    def __init__(self, save_path=None, load_path = None, dim=50, **kwargs):
        super().__init__(save_path=save_path, load_path=load_path)
        self.nclasses = kwargs['nclasses']

    def fit(self, *args):
        labels = list(set([a[0] for a in args[0]]))
        self.labelsdict = dict(zip(labels, range(len(labels))))
        #self.nclasses = len(labels)
        
    def save(self, *args, **kwargs):
        pass
        
    def load(self, *args, **kwargs):
        pass
    
    @overrides
    def __call__(self, batch, mean=False, *args, **kwargs):
        ohlabels = [self.labelsdict[l[0]] for l in batch]
        return to_categorical(ohlabels,num_classes=self.nclasses)
    
    def _make_padded_sequenses(self, docs, max_length, w2v):
        tokens = [doc.split(' ') for doc in docs]
        vecs = [[w2v[t] if t in w2v else np.zeros(50) for t in ts] for ts in tokens]
        seqs = np.array([np.pad(np.vstack(v),mode = 'constant', pad_width = ((0,max_length-len(v)),(0,0))) if len(v)<max_length else np.vstack(v)[:max_length,:] for v in vecs])
        return seqs
    
    
        