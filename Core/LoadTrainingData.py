# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:55:37 2020

@author: Sergey
"""

import os,json, h5py , csv, copy
import numpy as np, tensorflow as tf




class DataLoader():
    def __init__(self,data_train):
        
        json_file = open(os.path.join(data_train,'sim_params.json'))
        self.params = json.load(json_file)
        self.data_folder = data_train


    def load_data(self):


        X_Data = np.array([], dtype=np.float32).reshape(0,self.params["FragmentSize"])
        Y_Data = np.array([], dtype=np.float32).reshape(0,1)



        for folder in os.listdir(self.data_folder):
            if os.path.isdir(os.path.join(self.data_folder,folder)):
                for file in os.listdir(os.path.join(self.data_folder,folder)):
                    if '.npz' in file:
                        npzfile = np.load(os.path.join(self.data_folder,folder,file ))
                        X_Data = np.vstack([X_Data, npzfile['X_data']/100])
                        if 'Random' in file:
                            Y_Data = np.vstack([Y_Data, np.expand_dims(np.zeros(npzfile['X_data'].shape[0]),-1)])
                        else:
                            Y_Data = np.vstack([Y_Data, np.expand_dims(np.ones(npzfile['X_data'].shape[0]),-1)])


        train_indeces = np.arange(0,X_Data.shape[0])
        np.random.shuffle(train_indeces)

        X_Data = X_Data[train_indeces]
        Y_Data = Y_Data[train_indeces]
        return X_Data, Y_Data


class ExperimentalData():
    def __init__(self):
        pass

    def load_experimental_data(self,data_folder):

        if 'hdf5' in data_folder:
            data = self._read_data(data_folder)
        if 'csv'  in data_folder:
            data = []
            with open(data_folder) as csvfile:
                datareader = csv.reader(csvfile)
                for row in datareader:
                    data.append([float(x) for x in row])

        return data



    def _traverse_datasets(self,hdf_file):
        def h5py_dataset_iterator(g, prefix=''):
            for key in g.keys():
                item = g[key]
                path = f'{prefix}/{key}'
                if isinstance(item, h5py.Dataset): # test for dataset
                    yield (path, item)
                elif isinstance(item, h5py.Group): # test for group (go down)
                    yield from h5py_dataset_iterator(item, path)
                    
        for path, _ in h5py_dataset_iterator(hdf_file):
            yield path



    def _read_data(self, file_name: str):
        data = {}
        maxima = []

        with h5py.File(file_name, 'r') as f:
            for dset in self._traverse_datasets(f):
                data[dset] = f[dset][()]
                groups = dset.split('/')
                if len(groups) >= 2 and groups[1] == "maxima":
                    maxima.append(f[dset][()])
        return maxima

def eval_models(cnn_folder,output_path,x_data,ths):
    
    ths = 1-ths
    outputs = []
    genomes = []
    for genome in os.listdir(cnn_folder):
        genomes.append(genome)
        if os.path.isdir(os.path.join(cnn_folder,genome)):
            with open(os.path.join(cnn_folder,genome, 'model-Architecture.json'), 'r') as json_file:
                
                json_savedModel= json_file.read()
            CNN_genome = tf.keras.models.model_from_json(json_savedModel)
            fragment_size = CNN_genome.layers[0].get_output_at(0).get_shape().as_list()[1]
            CNN_genome.load_weights(os.path.join(cnn_folder,genome, 'modelBestLoss.hdf5'))


            for id,data in enumerate(x_data): 
                if len(data)> fragment_size:
                    start_ind = np.random.randint(0,len(data)-fragment_size)
                    x_data[id] = data[start_ind:start_ind+fragment_size]

                if len(data)< fragment_size:
                    x_data[id] = []

            x_data = [x for x in x_data if len(x)>0]

            x_data_cnn_d = np.array(x_data) 
            x_data_cnn_f = np.fliplr(np.array(x_data))

            def_pred = CNN_genome.predict(np.expand_dims(x_data_cnn_d,-1))
            def_flip = CNN_genome.predict(np.expand_dims(x_data_cnn_f,-1))

            scores = np.max(np.hstack([def_pred,def_flip]),axis=-1)
            outputs.append(scores)

    outputs = np.array(outputs).T
    with open(os.path.join(output_path  , 'out.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(genomes)
        for row in outputs:  
            writer.writerow(row)

    thresholded = copy.deepcopy(outputs)
    thresholded[thresholded<=ths] = 0
    thresholded[thresholded>ths] = 1

 


    return genomes, thresholded
    


