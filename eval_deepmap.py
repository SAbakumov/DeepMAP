# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:55:37 2020

@author: Sergey
"""
import tensorflow as tf, argparse
from Nets.Training import *
from Core.LoadTrainingData import *
from Core.Misc import kbToPx , normalize_local







gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
	except RuntimeError as e:
		print(e)



parser = argparse.ArgumentParser(description='Evaluation of a CNN set on experimental/simulated data. Outputs a CSV file with rows: scores from the CNN, columns: corresponding genomes')
parser.add_argument('data_to_analyse', metavar='Data path', type=str,
                    help='Absolute path to experimental/simulated data. Can be either in hdf5 or .csv format. The rows in CSV data must correspond to traces.')
parser.add_argument('model_path',  metavar='Model path', type=str,
                    help='Path to the folder containing all models. Folder must contain sub-folders for each of the models. Within these subfolders, a file model-Architecture.json and modelBestLoss.hdf5 must be present.')
parser.add_argument('output_path',  metavar='Output path', type=str,
                    help='Absolute path for the output file')


args = parser.parse_args()


data_to_analyse = args.data_to_analyse#r'D:\Sergey\FluorocodeMain\CNN_matching\Data\Experimental\EColi\segmentation-results.hdf5'
cnn_folder      = args.model_path#r'D:\Sergey\FluorocodeMain\CNN_matching\Models\PaperModels'
output_path     = args.output_path


data_loader = ExperimentalData()
x_data = []

for data in data_loader.load_experimental_data(data_to_analyse):
	pixel_local_window = kbToPx(10000,[1.72,0.34,78.6])
	data = normalize_local(pixel_local_window, data)
	x_data.append(data*100)

	



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
   







				









