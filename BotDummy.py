#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:56:26 2018

@author: lsm
"""

# things we need for NLP
import numpy as np
import pickle
import json
import pandas as pd
import random
from intents.IntentsClassifier import IntentsClassifier
    
class BotDummy():
    
    def __init__(self, config_file):
        self.config= json.load(open(config_file,'r'))
        sub_configs = {
                'Оплата':'intents/subs/pay/pay_config.json'
                }
        self.classifier = IntentsClassifier(root_config_path = 'intents/root/cf_config.json', sub_configs = sub_configs)
        self.answers_pay = pd.read_csv(self.config['answers_pay'])
        self.answers_default = pd.read_csv(self.config['answers_default'])
        self.ERROR_THRESHOLD_ROOT = 0.5
        self.ERROR_THRESHOLD_SUB = 0.4
        
    def __eval_linmodels(self, message):
        cleaned = self.tn.transform([message])
        root = {
            'decision':self.lin_model_root.predict(cleaned)[0],
            'confidence': np.max(self.lin_model_root.predict_proba(cleaned))
        }
        sublevel = {}
        if root['decision'] in self.sublevel_models:
            sublevel['decision'] = self.sublevel_models[root['decision']].predict(cleaned)[0]
            sublevel['confidence'] = np.max(self.sublevel_models[root['decision']].predict_proba(cleaned))
        return {
            'root': root,
            'sub': sublevel}

    def __classify(self,sentence):
        pass
    
    def __response(self,sentence, userID='123', show_details=False):
        results = self.classify(sentence)
        print(results)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in self.__intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            if show_details: print ('context:', i['context_set'])
                            if not userID in self.context:
                                self.context[userID] = []
                            self.context[userID].append(i['context_set'])
    
                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                            (userID in self.context and 'context_filter' in i and i['context_filter'] in self.context[userID][-2:]):
                            if show_details: print ('tag:', i['tag'])
                            # a random response from the intent
                            return print(random.choice(i['responses']))
                            #print(random.choice(i['responses']))
    
                results.pop(0)
    def run(self,message):
        res = self.classifier.run(message)
        for dec in res['root']['decision']:
            if res['root']['confidence']>self.ERROR_THRESHOLD_ROOT:
                print('root category: %s (confidence: %4f)'%(dec,res['root']['confidence']))
                if dec in res['subs']:
                    for sd in res['subs'][dec]['decision']:
                        if res['subs'][dec]['confidence']>self.ERROR_THRESHOLD_SUB:
                            print('\t sub-category: %s (confidence: %4f)'%(sd,res['subs'][dec]['confidence']))
                            answer = self.answers_pay[self.answers_pay.category == sd].answer.tolist()[0]
                            print('\nAnswer: %s' % (answer))
                        else:
                            print('subcategory unknown')
                else:
                    answer = self.answers_default[self.answers_default.category == dec].answer.tolist()[0]
                    print('\nAnswer: %s'%(answer))
            else:
                print('unknown')
if __name__ == '__main__':
    bot = BotDummy('config.json')
    while True:
        msg = input()
        print('\n')
        bot.run(msg)