
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:59:10 2020

@author: Sergey
"""


import Core.Misc as Misc
import Core.SIMTraces as SIMTraces
import os,json


import Core.TraceGenerator as TraceGenerator
import argparse







def GenTraces(TraceGen, genome, transform, Params):        
    if genome!='Random':
        TraceGen.ObtainTraces(transform, genome)
        
    elif genome == 'Random':
        TraceGen.ObtainRandomTraces(Params["Random-max"],Params["Random-min"],Params["RandomLength"],genome,transform)




def CallTraceGeneration(Params):

    # json_file = open(args.sim_params)

    # Params = json.load(json_file)

    
    # Params['Genomes'] = args.genome_path
    # Params['Save_path']  = args.save_path
    # Params['ConcatToCsv']  = args.csv

    # if Params['Random']:

        # SIMTRC    = SIMTraces.TSIMTraces('Random',Params["StretchingFactor"],0.34,0,Params["Enzyme"],Params["PixelSize"],Params['PixelShift'] ,Params["FPR"],Params["FPR2"],Params["FragmentSize"],
        #             Params["Min#ShuffledFrags"],Params["Max#ShuffledFrags"],Params["MinLengthShuffledFrags"],Params["MaxLengthShuffledFrags"],-1)
        # Gauss      = Misc.GetGauss1d(Params["FragmentSize"] , Misc.FWHMtoSigma(Misc.GetFWHM(Params["Wavelength"],Params["NA"])),Params["PixelSize"] )
        # ReCutsInPx  = SIMTRC.GetGenome(Params,'Random' )
        # TraceGen   = TraceGenerator.TraceGenerator(SIMTRC, ReCutsInPx,Gauss,[],Params)
        # for t in range(Params["NumTransformations"]):
        #     GenTraces(TraceGen, 'Random', t, Params)    

    total_random_lags = 0

    for gen_id, genome in enumerate(os.listdir(Params["Genomes"])):
        print('Generating simulated data from genome:'+genome)
        SIMTRC    = SIMTraces.TSIMTraces(genome,Params["StretchingFactor"],0.34,0,Params["Enzyme"],Params["PixelSize"],Params['PixelShift'] ,Params["FPR"],Params["FPR2"],Params["FragmentSize"],
                    Params["Min#ShuffledFrags"],Params["Max#ShuffledFrags"],Params["MinLengthShuffledFrags"],Params["MaxLengthShuffledFrags"],gen_id)
        SIMTRC.set_db_path(os.path.join(Params["Genomes"], genome))
        Gauss      = Misc.GetGauss1d(Params["FragmentSize"] , Misc.FWHMtoSigma(Misc.GetFWHM(Params["Wavelength"],Params["NA"])),Params["PixelSize"] )
        ReCutsInPx  = SIMTRC.GetGenome(Params,genome)


        TraceGen   = TraceGenerator.TraceGenerator(SIMTRC, ReCutsInPx,Gauss,[],Params)


        for t in range(Params["NumTransformations"]):
            GenTraces(TraceGen, genome, t, Params)    
        total_random_lags+=len(TraceGen.SimTraces.Lags)*len(Params["StretchingFactor"])
            
    print('Generating random simulated fragments')

    SIMTRC    = SIMTraces.TSIMTraces('Random',Params["StretchingFactor"],0.34,0,Params["Enzyme"],Params["PixelSize"],Params['PixelShift'] ,Params["FPR"],Params["FPR2"],Params["FragmentSize"],
                Params["Min#ShuffledFrags"],Params["Max#ShuffledFrags"],Params["MinLengthShuffledFrags"],Params["MaxLengthShuffledFrags"],-1)
    Gauss      = Misc.GetGauss1d(Params["FragmentSize"] , Misc.FWHMtoSigma(Misc.GetFWHM(Params["Wavelength"],Params["NA"])),Params["PixelSize"] )
    ReCutsInPx  = SIMTRC.GetGenome(Params,'Random' )
    TraceGen   = TraceGenerator.TraceGenerator(SIMTRC, ReCutsInPx,Gauss,[],Params)
    Params["RandomLength"] = total_random_lags

    for t in range(Params["NumTransformations"]):
        GenTraces(TraceGen, 'Random', t, Params)      

    with open(os.path.join(Params['Save_path'],'sim_params.json'), 'w') as fp:
        json.dump(Params, fp)

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='Generation of training/validation or test data for optical mapping.')
    parser.add_argument('sim_params', metavar='params.json', type=str,
                        help='Path to simulation parameters as defined in params.json file')
    parser.add_argument('genome_path',  metavar='Genome path', type=str,
                        help='Path to reference genomes. Must be complete level or scaffold level assemblies in .fna or .fasta formats')
    parser.add_argument('save_path',  metavar='Save path', type=str,
                        help='Path to store training data. Store .npz arrays, and optionally .csv if flagged in params.json')
    parser.add_argument('--csv', nargs='?', const=True, type=bool, help='[Optional] Duplicate save as CSV')

    args = parser.parse_args()

    json_file = open(args.sim_params)

    Params = json.load(json_file)

    
    Params['Genomes'] = args.genome_path
    Params['Save_path']  = args.save_path
    Params['ConcatToCsv']  = args.csv

    CallTraceGeneration(Params)

    
    
    
    
    
    
