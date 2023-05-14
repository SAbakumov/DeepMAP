# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:30:11 2020

@author: Sergey
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np 
import os


def ConvBlock(x , num_kernel,kernel_size):
    x = layers.Conv1D(num_kernel,kernel_size,padding='same',activation='relu')(x)
    x = layers.Conv1D(num_kernel,kernel_size,padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    return x 



def UNet(input_shape):
    input = layers.Input(shape = (input_shape,1))
    # x = tf.cast(input, dtype=np.float32)
    x = tf.math.divide(input,1)

    num_kernels = [16, 32, 64]
    kernel_size = 3
    output_layers = []
    for num_kernel in num_kernels:
        x = ConvBlock(x , num_kernel,kernel_size)
        output_layers = [x] + output_layers  
        x = layers.MaxPooling1D(pool_size=2)(x)


    x = ConvBlock(x , num_kernels[-1]*2,kernel_size)
    # x = layers.Conv1DTranspose(512,kernel_size,strides = 2 ,padding='same',activation='relu')(x)

    num_kernels.reverse()
    for id, num_kernel in enumerate(num_kernels):
        x = layers.Conv1DTranspose(num_kernel,kernel_size,strides = 2 ,padding='same',activation='relu')(x)
        x = layers.Concatenate()([output_layers[id],x])
        x = ConvBlock(x , num_kernel,kernel_size)

        # x = layers.Conv1DTranspose(num_kernels[id],kernel_size,strides = 2 ,padding='same',activation='relu')(x)
    
    output = layers.Conv1D(1,kernel_size,padding='same',activation='linear')(x)
    # output = layers.Dense(units=classes, activation = 'softmax')(x)
    
    CNN = tf.keras.Model(input,output,name='CNN1D')
    return CNN






if __name__ == '__main__':
    model = UNet(512)
    model.summary()
    # model_json = model.to_json()
    # with open( "Vibrio-Architecture.json", "w") as json_file:
    #     json_file.write(model_json)