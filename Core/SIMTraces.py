# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:37:06 2020

@author: Sergey
"""
import sys,numpy as np
import os,csv
sys.path.insert(1, os.path.join(os.path.dirname(__file__)))

from Bio import SeqIO
import Bio
import Core.Misc as msc
import random

class TSIMTraces:
      def __init__(self, Species, Stretch, BPSize, Optics,Enzyme,PixelSZ,Shift,FPR,FPR2,frag_size, Min_Shuffled_Frags, Max_Shuffled_Frags, Min_Length_Shuffled_Frags, Max_Length_Shuffled_Frags,gen_id):
        self.Species = Species
        self.Stretch = Stretch
        self.BPSize = BPSize
        self.Optics = Optics
        self.Enzyme = Enzyme
        self.PixelSize = PixelSZ
        self.PixelShift = Shift
        self.FPR = FPR
        self.FPR2 = FPR2
        self.Trace = []
        self.RandomTraces = []
        self.Map = []
        self.frag_size = frag_size
        self.Min_Shuffled_Frags = Min_Shuffled_Frags
        self.Max_Shuffled_Frags = Max_Shuffled_Frags
        self.Min_Length_Shuffled_Frags = Min_Length_Shuffled_Frags
        self.Max_Length_Shuffled_Frags = Max_Length_Shuffled_Frags
        self.RefProfile = {}
        self.gen_id = gen_id

        ROOT_DIR = os.path.abspath(os.curdir)
        self.DataBasePath = os.path.join(ROOT_DIR, 'DataBases')

      def set_db_path(self, db_path):
          self.DataBasePath = db_path

      def set_stretch(self, stretch):
          self.Stretch = stretch
        
      def set_recuts(self,recuts,gauss):
          self.recuts = recuts
          if self.Stretch not in self.RefProfile.keys():
            FullTrace = self.GetFullProfileIdeal(self.GetEffLabelingRate([self.recuts] ,1))
            self.RefProfile[self.Stretch]= self.GetFluorocodeProfile(FullTrace,gauss)[0]
      
      def set_region(self, *args):
          if type(args[1])==list:
            shft = np.random.randint(args[0][1],args[0][2])
            self.region = [args[0],args[0]+shft]
          else:
            shft = np.random.randint(0,args[2])
            self.region = [args[0]+shft,args[0]+shft+args[1]]
          
      def set_lags(self,step):
          # if FromLags:
          #     if len(Lags)!=0:
          #       Lags = [x for x in Lags if x<len(self.RefProfile[self.Stretch])-self.frag_size-20]
          #       self.Lags = Lags
          #     else:
          #       self.Lags = ( np.random.randint(0,len(self.RefProfile[self.Stretch])-self.frag_size-5,size=10000)).tolist()
          # else:

          self.Lags = list( range(20,len(self.RefProfile[self.Stretch])-self.frag_size-20,step))
              # random.shuffle(self.Lags)

      def get_lags(self,FromLags,Lags,step):
          if FromLags:
              Lags = [x for x in Lags if x<len(self.RefProfile[self.Stretch])-self.frag_size-20]
              return Lags
          else:
              return list( range(20,len(self.RefProfile[self.Stretch])-self.frag_size-20,step))


      def set_labellingrate(self,Up,Low):
          self.Up = Up
          self.Low = Low
        
      def get_labellingrate(self):
          labelrate = np.random.uniform(self.Low,self.Up)
          return labelrate
      
      def get_EffLabelledProfile(self):
          self.FullTrace = self.GetFullProfile(self.GetEffLabelingRate([self.recuts] ,self.get_labellingrate()))
        
      def get_FPR(self):
          self.FullTrace = self.YieldFPR(self.FullTrace)
          
      def get_WrongRegions(self):
          self.FullTrace = self.YieldWrongRegions(self.FullTrace)
      def get_FluorocodeProfile(self,gauss):
          trc = self.FullTrace[self.region[0]:self.region[1]]
          conv_trace = self.GetFluorocodeProfile([trc],gauss)
          return conv_trace
        
          
      
        
      
        
      def GetTraceRestrictions(self):
    
        
        genome = SeqIO.parse(os.path.join(self.DataBasePath), "fasta")

        CompleteSequence = []
        for record in genome:
            CompleteSequence.append(record.seq)
        CompleteSequence = "".join([str(seq_rec) for seq_rec in CompleteSequence])
            
        cuts =  msc.rebasecuts(self.Enzyme,Bio.Seq.Seq(CompleteSequence) )
        
        
        return cuts
    
    
      def GetEffLabelingRate(self,Traces,LabelingEfficiency):
        EffLabeledTraces = []
        for trc in Traces:
            EffLabeledTraces.append(np.random.choice(trc,np.int(LabelingEfficiency*len(trc)),replace=False))

        return EffLabeledTraces
    
    
      def GetDyeLocationsInPixel(self,ReCuts,strtch):
        ReCuts = np.array(ReCuts)
        # ReCuts = ReCuts-ReCuts[0]        
        ReCutsInPx = msc.kbToPx(ReCuts,[strtch, self.BPSize,self.PixelSize])
          
        return ReCutsInPx
    
      def GetTraceProfile(self,trace,gauss,size,orrarr):
        x = np.zeros(size)
        trace = trace-np.min(trace)
        for i in range(0,len(trace)):
            try:
                x[int(np.round(trace.item(i)))] = x[int(np.round(trace.item(i)))]+1+random.uniform(-0.2,0.2)
            except:
                continue
            
        signal = np.convolve(x,gauss, mode = 'same')
        signal = msc.ZScoreTransform(signal)
        return signal
    
      def GetFluorocodeProfile(self,trace,gauss):
        signals = []
        for trc in trace:
            signals.append(np.convolve(trc,gauss, mode = 'same'))
        return signals
    
    

        # return trace
    
      def GetFullProfile(self,genome):
        genome = genome[0]

        genome =(genome+ np.random.uniform(-self.PixelShift,self.PixelShift,size = genome.shape)).astype(int)
        Trace = np.zeros((np.max(genome)+10).item())
        u, c = np.unique(genome, return_counts=True)
        # c =  c*np.random.gamma(self.AmplitudeVariation[0],self.AmplitudeVariation[1],size = c.shape)
        # c =  c+c*np.random.uniform(-0.2,0.2,size = c.shape)
        # c = self.GetDyeAmp(c,-0.2,0.2)
        Trace[u]=Trace[u]+ c
        return Trace
        
      def GetDyeAmp(self,c):
          # c =  c+c*np.random.uniform(-0.2,0.2,size = c.shape)
        #   c = c
          c = c* np.random.gamma(self.AmplitudeVariation[0],self.AmplitudeVariation[1],size =c.shape)
          return c

      def YieldFPR(self,trc):
        if self.FPR>0:
            num_dyes = int( msc.PxTokb(self.frag_size, self)*self.FPR/1000)
            fpr_locs = np.random.uniform(self.region[0],self.region[1],num_dyes )
            # fpr_amps = self.GetDyeAmp(np.ones(fpr_locs.shape),-0.2,0.2)
            fpr_amps = np.ones(fpr_locs.shape)

            # fpr_amps = np.random.gamma(self.AmplitudeVariation[0],self.AmplitudeVariation[1],size = num_dyes)
            trc[(fpr_locs).astype(np.int64)] =  trc[(fpr_locs).astype(np.int64)]+fpr_amps

            if self.FPR2>0:
                num_dyes2 = int( msc.PxTokb(self.frag_size, self)*self.FPR2/1000)
                fpr_locs = np.random.uniform(self.region[0],self.region[1],num_dyes2)
                # fpr_amps = 2*np.random.gamma(self.AmplitudeVariation[0],self.AmplitudeVariation[1],size = num_dyes2)
                fpr_amps = 2*np.ones(fpr_locs.shape)

                trc[(fpr_locs).astype(np.int64)] =  trc[(fpr_locs).astype(np.int64)]+fpr_amps
    

        return trc
      def YieldWrongRegions(self,trc):
          if self.Min_Shuffled_Frags!=self.Max_Shuffled_Frags:
            numregs = np.random.randint(self.Min_Shuffled_Frags,self.Max_Shuffled_Frags) # np.random.randint(3,7)
          else:
            numregs = self.Min_Shuffled_Frags
          
          # trc , pxref = self.YieldInsertions(trc)
          # trc , pxref = self.YieldDeletions(trc, pxref)
          for i in range(numregs):
              startind = self.region[0]+ np.random.randint(0,self.frag_size)
              endind = startind + np.random.randint(self.Min_Length_Shuffled_Frags,self.Max_Length_Shuffled_Frags)
              trc[startind:endind] = np.random.permutation(trc[startind:endind])
          return trc 
              
      def YieldDeletions(self,trc,pxref):
          startind = self.region[0]+ np.random.randint(0,self.frag_size)
        #   endind = startind + np.random.randint(self.Min_Length_Shuffled_Frags,self.Max_Length_Shuffled_Frags)
          length = np.min([np.random.exponential(5000),10000])
          deletion = msc.kbToPx(length,[self.Stretch, self.BPSize,self.PixelSize])
          trc = np.delete(trc, np.arange(startind, startind+int(deletion)))

          pxref = np.delete(pxref,np.arange(startind-self.region[0], startind-self.region[0]+int(deletion)))
          pxref = pxref[0:self.frag_size]

          return trc , pxref


      def YieldInsertions(self,trc):
          startind = self.region[0]+ np.random.randint(0,self.frag_size)
          length = np.random.exponential(5000)
          insert = msc.kbToPx(length,[self.Stretch, self.BPSize,self.PixelSize])

          insert = np.zeros(int(insert))
          dyes   = np.random.choice(np.arange(0,len(insert)),int(length/1000*3))
          for dye in dyes:
              insert[dye]+=1
          trc = np.insert(trc, startind, insert)
          
          pixelwise_reference = np.ones(2*self.frag_size)
          pixelwise_reference[startind-self.region[0]:startind-self.region[0]+len(insert)] = 0


          return trc, pixelwise_reference


      def YieldNonLinearStretch(self,trc,region,frag_size):
              
          startind = region
          endind = startind + frag_size
          
          numzeros = np.random.randint(0,5)
          regind = np.random.randint(0,endind-numzeros, numzeros)

          arr = trc[startind:endind]
          arr = np.insert(arr,regind,0)
          
          trc[startind:endind] = arr[0:frag_size]
          return trc
                            
          
          
          
      def GetFullProfileIdeal(self,genome):
        # genome = genome[0]
        Traces = []
        for gen in genome:
            Trace = np.zeros([int(np.round(np.max(gen)).item())])
            for i in range(0,len(gen)):
                try:
                    # pos = int(np.round(genome[i].item()+np.random.uniform(-1.5,1.5)))
                    pos = int(np.round(gen[i].item()))
    
                    # Trace[pos] =  Trace[pos]+1+np.random.uniform(-0.1,0.1)
                    Trace[pos] =  Trace[pos]+1
                except:
                    # pos = int(np.round(genome[i].item()+np.random.uniform(-2,2)))
    
                    pos = int(np.round(gen[i].item()))
            
            Traces.append(Trace)
        return Traces
    



      def GetGenome(self,Params,genome):
        # ReCuts = []
        
        if not 'Random' in genome and ".csv" not in genome:
            
          ReCuts     = self.GetTraceRestrictions()
          ReCutsInPx = []
          for strtch in Params["StretchingFactor"]:
              ReCutsInPx.append(self.GetDyeLocationsInPixel(ReCuts,strtch))
              

        elif 'Random' in genome:
            ReCutsInPx = [0]
      

        return ReCutsInPx  





