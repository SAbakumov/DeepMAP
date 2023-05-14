# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:04:34 2020

@author: Boris
"""


import random,numpy as np
import time
import os
import csv
from tqdm import tqdm 
from Core import Misc


class TraceGenerator():
    def __init__(self, SIMTRC, ReCutsInPx, Gauss, NoiseProfiles,Params):
        self.SimTraces = SIMTRC
        self.Gauss = Gauss
        self.ReCutsInPx = ReCutsInPx
        self.Noise = NoiseProfiles
        self.Params = Params
        for key, value in Params.items():
            setattr(self, key, value)

        self.ToAddLabeled = []
        self.ToAddRef = []
        self.ToAddLabels = []
        self.Positions = []
      
            
    def reset(self):
        self.ToAddLabeled = []
        self.ToAddRef = []
        self.ToAddLabels = []

                
                
     
        
        
    def ObtainTraces(self,batchnum,genome):
        t = time.time()
        
        # EffLabeledTraces =[]
        # ReferenceData   = []
        # LabeledData     = []


        # if self.Params["Downsample"]>1:
        #     filter_setup = butter(3,1/self.Params["Downsample"], btype='low',output='sos')
        self.genome = self.SimTraces.Species
        self.generated_x = np.array([], dtype=np.float32).reshape(0,self.SimTraces.frag_size )
        self.generated_y = np.array([], dtype=np.int64).reshape(0,self.SimTraces.frag_size )

        for i in range(0,len(self.StretchingFactor)):
            self.SimTraces.set_stretch(self.StretchingFactor[i])
            self.SimTraces.set_recuts(self.ReCutsInPx[i],self.Gauss)
            self.SimTraces.set_labellingrate(self.LowerBoundEffLabelingRate, self.UpperBoundEffLabelingRate)
            self.SimTraces.set_lags(self.step)
            
            select_offsets = np.random.choice(self.SimTraces.Lags, int(self.PercentageChangedFrags*len(self.SimTraces.Lags)))

            generated_traces = np.zeros([len(self.SimTraces.Lags),self.SimTraces.frag_size ])
            pixelwise_traces = np.zeros([len(self.SimTraces.Lags),self.SimTraces.frag_size ])

            for trace_id, offset in enumerate(tqdm(self.SimTraces.Lags)):
                self.SimTraces.set_region(offset,self.FragmentSize,self.step)
                self.SimTraces.get_EffLabelledProfile()
                self.SimTraces.get_FPR()
                if offset in select_offsets:
                    self.SimTraces.get_WrongRegions()
                self.SimTraces.PixRegs = np.ones(self.SimTraces.frag_size)

              
                trc = self.SimTraces.get_FluorocodeProfile(self.Gauss)[0]
                
                
                trc = np.squeeze(trc+self.NoiseAmp*np.random.uniform(0,1,self.FragmentSize))
                trc = Misc.GetLocalNorm(trc,i,self.Params,self.SimTraces)

                generated_traces[trace_id,:] = trc 
                pixelwise_traces[trace_id,:] = self.SimTraces.PixRegs 

            self.generated_x = np.vstack([self.generated_x, generated_traces])
            self.generated_y = np.vstack([self.generated_y, pixelwise_traces])

        self.StoreData(batchnum)



            

    def StoreData(self,batchnum):
        genome = self.genome
        for ext in ['.fasta','.fna']:
            if ext in self.genome :
                genome = self.genome.replace(ext,'')

        if not os.path.exists(os.path.join(self.Save_path, genome)):
            os.makedirs(os.path.join(self.Save_path, genome))

        if self.ConcatToCsv:
            with open(os.path.join(self.Save_path, genome,genome+'_'+ str(batchnum)+'.csv'), 'w',newline='') as f:
                writer = csv.writer(f)
                for row in  self.generated_x:
                    writer.writerow(row)

        np.savez(os.path.join(self.Save_path, genome,genome+'_'+ str(batchnum)+'.npz'),**{'X_data': self.generated_x, 'Y_data': self.generated_y})
        


                
        
    def ObtainRandomTraces(self,maxNumDyes,minNumDyes, numprofiles,genome,batchnum):
    #   RandomTraces = []
    #   RandomLabels = []
    #   positions = []
      self.genome = 'Random'
      minNumDyesTotal = minNumDyes* Misc.PxTokb(self.FragmentSize, self.SimTraces)
      maxNumDyesTotal = maxNumDyes* Misc.PxTokb(self.FragmentSize, self.SimTraces)


      self.generated_x = np.array([], dtype=np.float32).reshape(0,self.SimTraces.frag_size )
      self.generated_y = np.array([], dtype=np.int64).reshape(0,self.SimTraces.frag_size )
      for i in range(0,len(self.StretchingFactor)):
            generated_traces = np.zeros([numprofiles,self.SimTraces.frag_size ])
            pixelwise_traces = np.zeros([numprofiles,self.SimTraces.frag_size ])
            for offset in tqdm(range(0,numprofiles)):
                # print(str(offset) + " out of " + str(numprofiles))

                numDyes =  np.random.randint(minNumDyesTotal, maxNumDyesTotal) 
                trace   =  np.zeros([self.FragmentSize])
                pos = np.random.uniform(0,self.FragmentSize,numDyes)
                u, c = np.unique(pos.astype(np.int16), return_counts=True)
                # c = c* np.random.gamma(self.amplitude_variation[0],self.amplitude_variation[1],size = c.shape)

                trace[u] = trace[u]+ c
                
                trace = self.SimTraces.GetFluorocodeProfile([trace],self.Gauss)[0]
                trc =  np.squeeze(trace+self.NoiseAmp*np.random.uniform(0,1,self.FragmentSize))
                # plt.figure(figsize=(20,4))
                # plt.plot(trace,color='blue')
                # plt.savefig("trace_example.svg")

                trc = Misc.GetLocalNorm(trc,i,self.Params,self.SimTraces)

                generated_traces[offset,:] = trc 
                pixelwise_traces[offset,:] = np.zeros(self.SimTraces.frag_size)

            self.generated_x = np.vstack([self.generated_x, generated_traces])
            self.generated_y = np.vstack([self.generated_y, pixelwise_traces])
      self.StoreData(batchnum)

              

    
