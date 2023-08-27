import numpy as np  , os, sys,csv , tqdm
sys.path.insert(1, 'Core')
from SIMTraces import TSIMTraces 
from LoadTrainingData import ExperimentalData
import Misc as Misc

class XCorrAlignment(TSIMTraces , ExperimentalData):
    def __init__(self, genome_folder, stretch_factors, pixel_size, wavelength, NA):
        self.genome_folder = genome_folder 
        self.stretch_factors = stretch_factors  

        self.NA = NA  
        self.Wavelength = wavelength
        self.BPSize = 0.34
        self.PixelSize = pixel_size
        self.pixel_local_window = Misc.kbToPx(10000,[1.72,0.34,self.PixelSize])
        self.Enzyme = 'TaqI'
        self.Gauss = Misc.GetGauss1d(100 , Misc.FWHMtoSigma(Misc.GetFWHM(self.Wavelength,self.NA)),self.PixelSize)

    def generate_genomes(self):
        re_cuts = {}
        for genome_fasta  in os.listdir( self.genome_folder):
            if '.fasta' in genome_fasta or '.fna' in genome_fasta:
                self.DataBasePath = os.path.join(self.genome_folder,genome_fasta)
                re_cuts[genome_fasta] = self.GetGenome(self.stretch_factors, os.path.join(self.genome_folder,genome_fasta))
                re_cuts[genome_fasta] = [self.GetFullProfileIdeal([g])[0] for g in re_cuts[genome_fasta] ]

                re_cuts[genome_fasta] = self.GetFluorocodeProfile(re_cuts[genome_fasta],self.Gauss)
                re_cuts[genome_fasta] = [ Misc.normalize_local(self.pixel_local_window, data) for data in  re_cuts[genome_fasta]]
        self.reference_genomes = re_cuts

    def load_data(self,data_folder):
        data_loader = ExperimentalData()
        self.x_data = []

        for data in data_loader.load_experimental_data(data_folder):
            data = Misc.normalize_local(self.pixel_local_window, data)
            self.x_data.append(data)

    def align(self):
        self.alignment_matrix = np.zeros([len(self.x_data), len(self.reference_genomes)])

        for gen_id , genome in enumerate(self.reference_genomes.keys()):
            reference = self.reference_genomes[genome]
            for id, trace in enumerate(self.x_data):
                print(id)
                scores_per_stretch = []
                max_scores_per_stretch = []
                for ref_strtch in reference: 
                    scores = np.correlate(trace/len(trace), ref_strtch,mode='same')
                    score_flipped = np.correlate(np.flipud(trace)/len(trace), ref_strtch,mode='same')

                    sc = np.max(scores)
                    sc_fl = np.max(score_flipped)

                    dir = np.argmax([sc, sc_fl])
                    if dir==0:
                        scores = scores 
                        sc = sc   
                    if dir==1:
                        scores = score_flipped
                        sc = sc_fl 


                    q25 = np.quantile(scores, 0.25)
                    q75 = np.quantile(scores, 0.75)
                    max_score = np.max(scores)
                    q_score = -(max_score-q75)/(q75-q25)
                    # score rescaling to be compatible with CNN's score output
                    scores_per_stretch.append(10**q_score)
                    max_scores_per_stretch.append(max_score)


                score = np.min(scores_per_stretch)
                self.alignment_matrix[id, gen_id] = score


    def store_results(self, save_folder):
        with open(save_folder, 'w', newline='') as csvfile:
            header = []
            for genome in self.reference_genomes.keys():
                header.append(os.path.splitext(genome)[0])
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for row in self.alignment_matrix:
                writer.writerow(list(row))
                

if __name__ == '__main__': 
    db_folder = 'Data/ref_genomes_xcorr'
    tested_traces = 'Data/measured/MixVibrioEColiSalmonella/segmentation-results.hdf5'
    save_folder = 'Data/measured/MixVibrioEColiSalmonella/xcorr_results.csv'
    stretch_factor = np.arange(1.68, 1.77, 0.01)
    pixel_size = 78.6  
    wavelength = 590 
    NA = 1.49

    alignment = XCorrAlignment(db_folder, stretch_factor, pixel_size, wavelength, NA) 
    alignment.generate_genomes()
    alignment.load_data(tested_traces)
    alignment.align()
    alignment.store_results(save_folder)
    print('Done')

                






    