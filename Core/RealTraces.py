#%%
import h5py,sys,os,copy, csv,pandas as pd, numpy as np
sys.path.insert(1, os.path.join(os.path.dirname(__file__),"Core"))

import Misc
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

def traverse_datasets(hdf_file):
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





#%%

class RealTraces():
    def __init__(self):
        self.hdf_reader = HdfTraceReader()
        self.alignment = []
    def set_tracepath(self,tracepath):
        self.tracepath = tracepath
    def set_alignmentpath(self,alignmentpath):
        self.alignmentpath = alignmentpath
    def set_alignmentparams(self, alignmentparamspath):
        self.alginmentparamspath = alignmentparamspath
        df =pd.read_csv(self.alginmentparamspath, header=None)
        self.alignmentparams = df.to_dict('index')
        params =[]
        for key in list(self.alignmentparams.keys()):
            param = {self.alignmentparams[key][0]: self.alignmentparams[key][1]} 
            params.append(param)
        d = {}
        for dictionary in params:
            d.update(dictionary)
        self.alignmentparams =   d
    



    def load_traces_from_csv(self,filename, amount_of_traces, dtype=np.float32):
        data = []
        data_counter = 0
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
            for row in reader:  # each row is a list
                # if amount_of_traces <= 0, we load all the traces from the file
                if row is None or row == "" or row == [] or amount_of_traces > 0 and data_counter >= amount_of_traces:
                    break
                row = [x for x in row if x != '']
                if row == []:
                    break
                data.append(np.array(row).astype(dtype=dtype))
                data_counter = data_counter + 1
        return data

    def load_tested_traces(self):
        file_path = self.tracepath
        if file_path[-4:] == ".csv":
            self.tested_traces = self.load_traces_from_csv(file_path, sys.maxsize)
        elif file_path[-5:] == ".hdf5":
            self.hdf_reader.read_data(file_path)
            self.tested_traces = self.hdf_reader.maxima
        else:
            raise Exception("File format of the {0} file is not supported!!!".format(file_path))

    def load_alignment_file(self):
        with open(self.alignmentpath,'r') as csvfile:
            alignment = csv.reader(csvfile) 
            fields = next(alignment)
            alignment_per_genome = dict()
            for row in alignment:
                for genome in range(3,len(row)-11+1,11):
                    if row[genome] in alignment_per_genome.keys():
                        dict_row  = dict(zip(['tested_trace_id','xcorr','stretch','pvalue','direction','match_len'],[[int(row[0])],[float(row[genome+1])],[float(row[genome+2])],[float(row[genome+6])],[row[genome+4]],[int(row[genome+5])]]))
                        alignment_per_genome[row[genome]] = {key: alignment_per_genome[row[genome]][key]+dict_row[key]  for key in alignment_per_genome[row[genome]]}
                    else:
                        alignment_per_genome[row[genome]] = dict(zip(['tested_trace_id','xcorr','stretch','pvalue','direction','match_len'],[[int(row[0])],[float(row[genome+1])],[float(row[genome+2])],[float(row[genome+3])],[row[genome+4]],[int(row[genome+5])]]))

        self.alignment = alignment_per_genome


    def get_cut_length(self,length):
        to_remove_ids = []
        for alignment in self.alignment[list(self.alignment.keys())[0]]:
            for match_len_idx in range(0,len( self.alignment[list(self.alignment.keys())[0]]['match_len'])):
                if self.alignment[list(self.alignment.keys())[0]]['match_len'][match_len_idx]<=length:
                    to_remove_ids.append(self.alignment[list(self.alignment.keys())[0]]['tested_trace_id'][match_len_idx])
        
        for alignment in self.alignment:
            for field in self.alignment[alignment]:
                self.alignment[alignment][field] = [ self.alignment[alignment][field][idx] for idx in range(0,len( self.alignment[alignment][field])) if idx not in to_remove_ids]
    
    
    def get_cut_xcorr(self,xcorr,genome):
        to_remove_ids = []
        for match_len_idx in range(0,len( self.alignment[list(self.alignment.keys())[0]]['xcorr'])):
            if self.alignment[genome]['xcorr'][match_len_idx]<xcorr:
                to_remove_ids.append(self.alignment[genome]['tested_trace_id'][match_len_idx])
    
        for field in self.alignment[genome]:
            self.alignment[genome][field] = [ self.alignment[genome][field][idx] for idx in range(0,len( self.alignment[genome][field])) if idx not in to_remove_ids]

    def get_cut_pvals(self,pvalue,genome):
        to_remove_ids = []
        for match_len_idx in range(0,len( self.alignment[list(self.alignment.keys())[0]]['pvalue'])):
            if self.alignment[genome]['pvalue'][match_len_idx]<=pvalue:
                to_remove_ids.append(self.alignment[genome]['tested_trace_id'][match_len_idx])
    
        for field in self.alignment[genome]:
            self.alignment[genome][field] = [ self.alignment[genome][field][idx] for idx in range(0,len( self.alignment[genome][field])) if idx not in to_remove_ids]
    
    
    def get_xcorr_vals(self,genome):
        xcorr_pair = []
        for match_len_idx in range(0,len( self.alignment[list(self.alignment.keys())[0]]['xcorr'])):
            # if self.alignment[genome]['xcorr'][match_len_idx]<xcorr:
            #     to_remove_ids.append(self.alignment[genome]['tested_trace_id'][match_len_idx])
            xcorr_pair.append([self.alignment[genome]['xcorr'][match_len_idx],self.alignment[genome]['match_len'][match_len_idx]])
        return xcorr_pair


    def normalize_traces_local(self,window,normalize,pixelsz):
        if pixelsz > 0:
            self.normalized_traces = []
            for trace in self.tested_traces:
                if normalize:
                    trace  = Misc.normalize_local(Misc.kbToPx(window,[1.71,0.34,pixelsz]),trace)*100
                self.normalized_traces.append(trace)

    
    
        else:
            self.normalized_traces = []
            for trace in self.tested_traces:
                if normalize:
                    trace  = Misc.normalize_local(Misc.kbToPx(window,[1.71,0.34,float(self.alignmentparams['Pixel size in nanometers (image pixel size (final) [nm])'])]),trace)*100
                self.normalized_traces.append(trace)
    def normalize_traces_local_from_params(self,window,normalize,params):
        self.normalized_traces = []
        for trace in self.tested_traces:
            if normalize:
                trace  = Misc.normalize_local(Misc.kbToPx(window,[1.71,0.34,float(params["pixel_size"])]),trace)*100

            self.normalized_traces.append(trace)

    def pick_traces_subset(self,genome,fromalignment):
        self.trace_subset = []
        if fromalignment=="Alignment":
            for idx in self.alignment[genome]['tested_trace_id']:
                self.trace_subset.append(self.normalized_traces[idx])
        else:
            for trace in self.normalized_traces:
                self.trace_subset.append(trace)
              
    def Downsample(self,downsample):
        subsampled_traces  = []
        for trace in self.trace_subset:
            win = np.ones(downsample)/downsample # averaging window

            trace= np.convolve(trace, win, mode='same')
            trace = trace[::downsample]
            subsampled_traces.append(trace)
        self.trace_subset = subsampled_traces
        # return trc

    def prepare_input_size(self,sz):
        self.default_input_traces = []
        self.flipped_input_traces = []
        self.original = []
        for trace in self.trace_subset:
            if len(trace)>=sz:
            
                startind = np.random.randint(0,len(trace)-sz+1)
                self.default_input_traces.append(trace[startind:startind+sz])
                self.flipped_input_traces.append(np.flipud(trace[startind:startind+sz]))
                self.original.append(trace)
        self.flipped_input_traces= np.reshape(np.array(self.flipped_input_traces),[len(self.flipped_input_traces),sz,1])
        self.default_input_traces= np.reshape(np.array(self.default_input_traces),[len(self.default_input_traces),sz,1])

        
        


    def plot_traces(self,N):
        plt.figure(figsize=(10,5))
        plt.plot(self.tested_traces[N],color='b',linewidth=1.5)
        plt.xlabel("Distance [px]")

    def split_tested_traces(self,sz,lag):
        self.split_traces = []
        for trace in self.tested_traces:
            if len(trace)>sz:
                for n in range(0,len(trace)-sz,lag):
                    self.split_traces.append(trace[n:n+sz])


    def SaveToCSV(self, folder,traces):
        
        with open(folder, 'wt', newline='') as f:
            csv_writer = csv.writer(f)


            for row in traces:
                csv_writer.writerow(row)


class HdfTraceReader:
    def __init__(self):
        pass
        
    def read_data(self, file_name: str):
        self.data = {}
        self.maxima = []
        with h5py.File(file_name, 'r') as f:
            for dset in traverse_datasets(f):
                self.data[dset] = f[dset][()]
                if dset.split('/')[1] == "maxima":
                    self.maxima.append(f[dset][()])
    
    def print_data(self):
        for key, value in self.data.items():
            print(key) #, str(value)[:50])


if  __name__ == '__main__':
    traces = RealTraces()
    traces.set_tracepath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiEthGly\segmentation-results.hdf5")
    traces.set_alignmentpath(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TestPoorAlignment\logs_nw\FILE1-all-reference-species-results.csv")
    traces.set_alignmentparams(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TestPoorAlignment\logs_nw\FILE0-classification-parameters.csv")

    traces.load_tested_traces()
    traces.load_alignment_file()

    # traces.split_tested_traces(512,10)
    # traces.SaveToCSV(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TileScans\NC_000913.3\EColiEthGly\SplitTraces512.csv",traces.split_traces)
    # traces.load_alignment_file()


    traces.get_cut_length(400)
    # traces.get_cut_pvals(0.0004,'NC_000913.3')
    traces.normalize_traces_local([],False)
    traces.pick_traces_subset('NC_000913.3','Alignment')
    Trace = traces.trace_subset
    # traces.pick_traces_subset('NC_000913.3')
    # traces.prepare_input_size(400)
    with open(r"D:\Sergey\FluorocodeMain\Fluorocode\Fluorocode\Data\TestPoorAlignment\segmented_traces_big_only.csv", "w", newline="") as outfile:

        writer = csv.writer(outfile)

        for trace in traces.trace_subset:

            writer.writerow(trace)

    # traces.plot_traces(43)

# %%
