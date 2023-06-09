

## Installation

Download and unpack the .git repository. Create a new environment and install the requirements from requirements.txt (pip install -r )


## Simulating data

To train a CNN for a desired genome, the training and validation datasets have to be simulated. The default parameters for the simulation are provided in sim_params.json file as follows

```
{   "Wavelength" : 586,           -> Emission wavelength of the fluorophore
    "NA" : 1.45,                  -> NA of the objective
    "FragmentSize" :316,          -> Fragment size in pixels (number of points)
    "PixelSize" : 78.6,           -> Calibrated pixel size of the instrument
    "Enzyme" : "TaqI",            -> Restriction/Methyltransferase enzyme used for labelling.   
    "NumTransformations"  : 1,    -> Number of simulations. This is the total number of times that a single genome is sampled.
    "StretchingFactor" :[1.72],   -> Stretch factor for the simulations. Default is 1.72
    "LowerBoundEffLabelingRate" : 0.75, -> Lower bound labelling rate. Labelling rate is uniformly distributed between lower and upper bound 
    "UpperBoundEffLabelingRate" : 0.9,  -> Upper bound labelling rate. 
    "step" :3,                          -> Sampling step across genome (stride) in number of pixels
    "PixelShift":0.2,                   -> Random shift in dye position (+- PixelShift)
    "NoiseAmp": [0.2],                  -> Relative SNR
    "LocalNormWindow":10000,            -> Local normalization window in kb. Set to 0 in order to disable. See http://bigwww.epfl.ch/sage/soft/localnormalization/#:~:text=The%20local%20normalization%20tends%20to,uneven%20illumination%20or%20shading%20artifacts.http://bigwww.epfl.ch/sage/soft/localnormalization/#:~:text=The%20local%20normalization%20tends%20to,uneven%20illumination%20or%20shading%20artifacts. 
    "Min#ShuffledFrags":0,              -> Minimal number of shuffeled regions within the simulated trace
    "Max#ShuffledFrags":3,              -> Maximal number of shuffeled regions within the simulated trace
    "MinLengthShuffledFrags": 35,       -> Minimal length of shuffeled regions
    "MaxLengthShuffledFrags": 55,       -> Maximal length of shuffeled regions
    "Random" : true,                    -> Wether to generate a random reference as background for the training. 
    "RandomLength" : 12000,             -> Number of random fragments. Must be approx ~50% of total data
    "FPR": 0.2,                         -> False positive rate /kb
    "FPR2": 0.02,                       -> Double false positive rate
    "Random-min": 1.2,                  -> Minimal number of dyes/kb for random genome
    "Random-max": 6.8                   -> Maximal number of dyes/kb for random genome
   }    
```
To simulate the data, simply call python generate_trainingdata.py with appropriate arguments. 


```
usage: generate_trainingdata.py [-h] [--csv [CSV]] params.json Genome path Save path

Generation of training/validation or test data for optical mapping.

positional arguments:
  params.json  Path to simulation parameters as defined in params.json file
  Genome path  Path to reference genomes. Must be complete level or scaffold level assemblies in .fna or .fasta formats
  Save path    Path to store training data. Store .npz arrays, and optionally .csv 

optional arguments:
  -h, --help   show this help message and exit
  --csv [CSV]  [Optional] Duplicate save as CSV
```
Once the data is simulated, we are ready for the training of our CNN-classifier! Don't forget to simulate both the training and validation data separately, in order to avoid overfitting.

## Training

In order to train the classifier, you have to parse the input folder with the simulated genomes. Simply call the train_deepmap.py with appropriate arguments:
```
usage: train_deepmap.py [-h] Training data path Validation data path CNN save path

Training of deep neural nets for species recognition using optical mapping data

positional arguments:
  Training data path    Absolute path to training data. Must contain sim_params.json from generation
  Validation data path  Absolute path to validation data. Must contain sim_params.json from generation
  CNN save path         Path to save CNN weights and architecture

optional arguments:
  -h, --help            show this help message and exit
```

Note: training saves the weights corresponding to the best validation loss and runs by default for 400 epochs. If you see no net improvement in the validation, you can stop the training by closing/stopping the script.

## Evalutation

Once the network(s) are trained, it can be applied to real data. Currently, the script accepts .hdf5 from proprietary software and .csv files. In case of the CSV file, the rows of the CSV must correspond to measured traces. To call the set of networks on the data, simply store the networks in a separate folder with the folder names corresponding to the genomes. Each of the folders must contain the modelBestLoss.hdf5 and model-Architecture.json, which are normally stored during training.

To call the evaluation and get the corresponding output scores, simply call the eval_deepmap.py with appropriate arguments:

````
usage: eval_deepmap.py [-h] Data path Model path Output path

Evaluation of a CNN set on experimental/simulated data. Outputs a CSV file with rows: scores from the CNN, columns: corresponding genomes

positional arguments:
  Data path    Absolute path to experimental/simulated data. Can be either in hdf5 or .csv format. The rows in CSV data must correspond to traces.
  Model path   Path to the folder containing all models. Folder must contain sub-folders for each of the models. Within these subfolders, a file model-
               Architecture.json and modelBestLoss.hdf5 must be present.
  Output path  Absolute path for the output file

optional arguments:
  -h, --help   show this help message and exit
````


## License

[MIT](https://choosealicense.com/licenses/mit/)