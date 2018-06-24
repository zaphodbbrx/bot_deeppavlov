#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:44:26 2018

@author: lsm
"""

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.infer import interact_model
from text_normalizer import *
from embedder import *
from augmenter import *
from CNN_model import *
from label_transformer import *

interact_model('tns_config.json')