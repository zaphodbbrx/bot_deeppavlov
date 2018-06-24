#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:28:38 2018

@author: lsm
"""

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.utils import expand_path, set_deeppavlov_root
from deeppavlov.core.common.file import read_json
from model.pipeline.text_normalizer import *
from model.pipeline.embedder import *
from model.pipeline.CNN_model import *

config = read_json('model/subs/pay/pay_config.json')
set_deeppavlov_root(config)
train_evaluate_model_from_config('model/subs/pay/pay_config.json')