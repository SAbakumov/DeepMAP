# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:05:48 2020

@author: Boris
"""


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

json_file = open(os.path.join('D:\Sergey\FluorocodeMain\FluorocodeMain\TrainingCurves' ,'2020-10-24\model-3Species-LeuvenParameters-EColi.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join('D:\Sergey\FluorocodeMain\FluorocodeMain\TrainingCurves', '2020-10-24\model-3Species-LeuvenParameters-EColi.h5' ))


