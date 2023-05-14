# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:55:37 2020

@author: Sergey
"""
import tensorflow as tf, argparse
from Nets.Training import *
from Core.LoadTrainingData import DataLoader
from Nets.InceptionV3 import *
from Nets.VGG16 import *
from Nets.ResNet50 import *
from Nets.CNN1D import *







gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6048)])
	except RuntimeError as e:
		print(e)

parser = argparse.ArgumentParser(description='Training of deep neural nets for species recognition using optical mapping data')
parser.add_argument('train_path', metavar='Training data path', type=str,
                    help='Absolute path to training data. Must contain sim_params.json from generation')
parser.add_argument('val_path',  metavar='Validation data path', type=str,
                    help='Absolute path to validation data. Must contain sim_params.json from generation')
parser.add_argument('net_save_path',  metavar='CNN save path', type=str,
                    help='Path to save CNN weights and architecture')

args = parser.parse_args()


Train_data_loader = DataLoader(args.train_path)
Val_data_loader = DataLoader(args.val_path)

X_Data, Y_Data = Train_data_loader.load_data(args.train_path)
X_Data_val, Y_Data_val = Val_data_loader.load_data(args.val_path)


opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model =CNN1D((Train_data_loader.params['FragmentSize'],1))
model.summary()

training_loop = TrainModel(model)
training_loop.compile(optimizer=opt)


save_metriccallb = SaveMetrics(args.net_save_path)
save_modelcallb  = SaveBestModel(args.net_save_path)

training_loop.fit((X_Data,Y_Data),validation_data = (X_Data_val,Y_Data_val), batch_size = 128, epochs=400, verbose = 1,callbacks = [save_metriccallb,save_modelcallb])
