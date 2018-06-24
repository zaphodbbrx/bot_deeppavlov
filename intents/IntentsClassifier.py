#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:32:43 2018

@author: lsm
"""

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.utils import expand_path, set_deeppavlov_root
from deeppavlov.core.common.file import read_json
import numpy as np
from deeppavlov.core.commands.infer import *
from model.pipeline.embedder import *
from model.pipeline.CNN_model import *
from model.pipeline.text_normalizer import *


class IntentsClassifier():
    def __init__(self, root_config_path, sub_configs = {}):
        self.__root_config = root_config_path
        if len(list(sub_configs.items()))>0:
            self.__sub_configs = sub_configs
        
        root_config = read_json(root_config_path)
        self.__root_model = build_model_from_config(root_config)
        
        self.__sub_models = {}
        for cl,conf in self.__sub_configs.items():
            sc = read_json(conf)
            self.__sub_models[cl] = build_model_from_config(sc)

    
    def __predict(self, model, input_text):
        rp = model.pipe[0][-1]([input_text])
        for i in range(1,len(model.pipe)-1):
            rp = model.pipe[i][-1](rp)
        res = model.pipe[-1][-1](rp, predict_proba = True)
        dec = proba2labels(res, 
                           confident_threshold = model.pipe[-1][-1].confident_threshold,
                           classes=model.pipe[-1][-1].classes)[0]
        return {
            'decision': dec,
            'confidence': np.max(res)
               }
    
    def train(path_to_config):
        pass
    
    def run(self,message):
        res = {}
        root_config = read_json(self.__root_config)
        root_model = build_model_from_config(root_config)
        root_res = self.__predict(root_model,message)
        res['root'] = root_res
        res['subs'] = {}
        for dec in root_res['decision']:
            if dec in list(self.__sub_configs.keys()):
                sc = read_json(self.__sub_configs[dec])
                sub_model = build_model_from_config(sc)
                res['subs'][dec] = self.__predict(sub_model,message)
        return res
    
if __name__ == '__main__':
    sub_configs = {
            'Оплата':'subs/pay/pay_config.json'
            }
    mes = ''
    while mes != 'q':
        ic = IntentsClassifier(root_config_path='root/cf_config.json',sub_configs = sub_configs)
        mes = input()
        print(ic.run(mes))