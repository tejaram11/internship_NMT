# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 12:47:23 2023

@author: TEJA
"""

import pickle
import seaborn as sea

with open('models/train_data_10.pkl','rb') as d:
    data_train=pickle.load(d)
    
print(data_train)