# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:55:53 2020

@author: Sergey
"""


from Bio import Restriction
import Core.SIMTraces as SIMTraces
import numpy as np, os

import json
import csv
import h5py






def GetGauss1d(size,sigma,pixelsz):
    x = np.linspace(-np.round(size/2),np.round(size/2), size+1)*pixelsz
    Gauss = np.exp(-np.power(x,2)/(2*np.power(sigma,2)))
    Gauss = Gauss[0:len(Gauss)-1]
    return Gauss




def GetGauss(sigma,pixelsz):
    x = np.linspace(-18,17)*pixelsz
    y = np.linspace(-18,17)*pixelsz
    
    xv,yv = np.meshgrid(x,y)
    Gauss = np.multiply(np.exp(-np.power(xv,2)/(2*np.power(sigma,2))),np.exp(-np.power(yv,2)/(2*np.power(sigma,2))))
    return Gauss


   

def GetFWHM(wavelength,NA):
    FWHM =  0.61*wavelength/(NA)
    return FWHM
    
def FWHMtoSigma(FWHM):
    sigma = FWHM/2.3548
    return sigma 

def SigmatoFWHM(Sigma):
    FWHM = Sigma*2.3548
    return FWHM
    
def rebasecuts(Enzyme, Strand):
    batch = Restriction.RestrictionBatch()
    batch.add(Enzyme)
    enzyme = batch.get(Enzyme)    
    
    Sites = enzyme.search(Strand)
    
    return Sites



def kbToPx(arr,args):
    if type(args)==list:
        stretch, nmbp, pixelsz = args[0] , args[1] ,  args[2]
    else:
        stretch, nmbp, pixelsz = args.Stretch , args.BPSize ,  args.PixelSize

        
    arr  =   (arr*stretch*nmbp)/pixelsz
    return arr

def PxTokb(arr,args):
    if type(args)==list:
        stretch, nmbp, pixelsz = args[0] , args[1] ,  args[2]
    elif type(args)==SIMTraces.TSIMTraces:
        stretch, nmbp, pixelsz = args.Stretch , args.BPSize ,  args.PixelSize
    else:
        print('Unsupported data type in kbToPx, aborting execution')
    
    if type(stretch)==list:
        stretch=stretch[0]
    arr  =   (arr/stretch/nmbp)*pixelsz/1000
    return arr
    

           
def GetLocalNorm(trace,i,Params,SimTraces):
    if Params["LocalNormWindow"]>0:
        conv = [Params["StretchingFactor"][i] , SimTraces.BPSize ,  SimTraces.PixelSize]

        trace  = np.round(normalize_local(kbToPx(Params["LocalNormWindow"],conv),trace)*100).astype(np.int16)
    # elif Params["ZNorm"]:
    #     trace = (((trace- np.mean(trace))/np.std(trace))*100).astype(np.int16)
    # elif Params["Norm"]:
    #     trace = (trace/np.std(trace)*100).astype(np.int16)
    else:
        trace = (trace*100).astype(np.int16)
     
    return trace          
                       


    
def normalize_local(npoints, trace):


    window = np.ones(np.min([np.round(npoints).astype(np.int64), len(trace)])) / np.min([np.round(npoints).astype(np.int64), len(trace)])
    trace = np.pad(trace, (len(window),len(window)), 'constant',constant_values=(trace[0],trace[-1]))
    local_mean = np.convolve(trace, window, 'same')
    out = trace - local_mean
    local_var = np.convolve(np.power(out, 2), window, 'same')
    if np.sum(local_var[local_var > 0]) > 0:
        local_var[local_var == 0] = np.min(local_var[local_var > 0]) # avoid division by zero in next step
        out = out / np.sqrt(local_var)
    out = out[len(window):len(out)-len(window)]
    return out
        





    
def WriteDataParams(savedir,Params):
    jsonfile = json.dumps(Params)
    f = open(os.path.join(savedir ,'Params' +'.json'),"w")
    f.write(jsonfile)
    f.close()
    
    f = open(os.path.join(savedir ,'Params'+'.csv'),"w")
    w = csv.writer(f)
    for key, val in Params.items():
        w.writerow([key, str(val)])        
    f.close()
        
    
    
