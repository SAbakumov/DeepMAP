# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:55:37 2020

@author: Sergey
"""

from Nets.Training import TrainModel
import os,json, h5py , csv, copy , tqdm
import numpy as np, tensorflow as tf, pandas as pd 
import matplotlib.pyplot as plt 
from Bio import SeqIO



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
                for file in tqdm.tqdm(os.listdir(os.path.join(self.data_folder,folder))):
                    if '.npz' in file:
                        npzfile = np.load(os.path.join(self.data_folder,folder,file ))
                        X_Data = np.vstack([X_Data, npzfile['X_data']])
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

class Evaluation():
    def __init__(self):

        self.thresholds = {}




    def eval_models(self,cnn_folder,output_path,x_data):
        outputs = []
        genomes = []
        for genome in os.listdir(cnn_folder):
            if os.path.isdir(os.path.join(cnn_folder,genome)):

                for file in os.listdir(os.path.join(cnn_folder,genome)):
                    if '.fasta' in file or '.fna' in file: 
                        genome_length = len(SeqIO.read(os.path.join(cnn_folder,genome,file), "fasta"))
                genomes.append(genome)

                
                with open(os.path.join(cnn_folder,genome, 'model-Architecture.json'), 'r') as json_file:
                    
                    json_savedModel= json_file.read()
                CNN_genome = tf.keras.models.model_from_json(json_savedModel)
                fragment_size = CNN_genome.layers[0].get_output_at(0).get_shape().as_list()[1]
                CNN_genome.load_weights(os.path.join(cnn_folder,genome, 'modelBestLoss.hdf5') , by_name = True, skip_mismatch = True)
               


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

        self.outputs = np.array(outputs).T
        with open(os.path.join(output_path  , 'out.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(genomes)
            for row in self.outputs:  
                writer.writerow(row)

    
        self.genomes = genomes  
        # self.results = np.array(thresholded)
    def load_xcorr_data(self,folder):
        self.outputs = pd.read_csv(folder,header=None).to_numpy()
        self.genomes = [x for x in self.outputs[0]]
        self.outputs = 1-np.float32(np.array(self.outputs[1:]))


    def threshold(self,t):
        self.results = (self.outputs>t)*self.outputs
        self.results[self.results!=0] = 1
        for ind_row in range(0, self.results.shape[0]):
            if len(np.where(self.results[ind_row,:]==1)[0])>1:
                self.results[ind_row,:] = 0

        trace_matches = dict(zip([x for x  in self.genomes],[self.results[:,x] for x in range(0,self.results.shape[1])]))
        return trace_matches
        
    def get_abundance(self):
        self.abundances = {}
        trace_matches = self.threshold(self.ths)

        for genome in self.genomes:
            
            unassigned_traces = np.sum(~self.results.any(1))/self.outputs.shape[0]
            tp_matches = np.sum(trace_matches[genome])/self.outputs.shape[0]
            fp_matches = np.sum([np.sum(trace_matches[key]) for key in trace_matches.keys() if key!=genome])/self.results.shape[0]

            self.abundances[genome] = [tp_matches, fp_matches,unassigned_traces, genome]

        total_matched = np.sum([self.abundances[gen][0]  for gen in self.abundances.keys()])
        for genome in self.abundances.keys():
            self.abundances[genome][0] =self.abundances[genome][0]/total_matched


    def eval_statistics(self,fp_val):


        self.match_statistics = {}
        self.fp_val = fp_val
        for genome in self.genomes:
            self.match_statistics[genome] =[0, 0,1, genome,0 ]

        # ths = [1-0.0005]

        ths = 1-np.logspace(-4,0,base=10,num=500)
        for t in ths:
            trace_matches = self.threshold(t)

            for genome in self.genomes:
                unassigned_traces =np.round(100* np.sum(~self.results.any(1))/self.outputs.shape[0])
                tp_matches = np.round(100*np.sum(trace_matches[genome])/self.outputs.shape[0])
                fp_matches = np.round(100*np.sum([np.sum(trace_matches[key]) for key in trace_matches.keys() if key!=genome])/self.results.shape[0])


                if fp_matches==fp_val*100 and tp_matches> self.match_statistics[genome][0]:

                    self.match_statistics[genome] = [tp_matches, fp_matches,unassigned_traces, 0, t ]
                


    def get_statistics(self,tp_genome):
        [tp_matches, fp_matches,unassigned_traces, genlen, ths ]  = self.match_statistics[tp_genome]
        self.ths= ths

        labels = [  'Ground truth','False positives','Unassigned']
        colors = [[163/255, 17/255, 7/255], [255/255, 126/255, 117/255], [128/255, 147/255, 255/255]]
        explode = (0.1, 0, 0)  # explode 1st slice
        plt.pie([tp_matches,fp_matches,unassigned_traces], explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140,wedgeprops={'alpha':0.4,"edgecolor":"k",'linewidth': 2})

        plt.axis('equal')

