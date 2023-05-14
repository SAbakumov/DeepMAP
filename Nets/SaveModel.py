# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:15:36 2020

@author: Boris
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

 
def SaveModel(model):
    model_json = model.to_json()
    with open("model-LagsOnlyEColiLaurensStep2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model-LagsOnlyEColiLaurensStep2.h5")
    print("Saved model to disk")
    
    

    