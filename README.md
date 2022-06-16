# Embedding neurophysiological signals

## Dependancies installation
To install the necessary dependancies, you can simply use `conda`:
```
conda env create -f environment.yml
```
Then you must activate it to be able to reproduce the experiments:
```
conda activate embedding_eeg_2022
```

## Reproduction of the results
Before starting anything, you must create a file named `local_config.yaml` that suits your hardware configuration. To do this, you can make use of the template named `local_config_TEMPLATE.yaml`.

Then, you can run the experiments.
The scripts named `fbcsp_within-subject.py`, `eegnet_linear_probing.py`, and `eegnet_cross-subject.py` will respectively run the MOABB evaluations for the **FBCSP**, **EEGNet+LP**, and **EEGNet** pipelines (see article).

Before running `eegnet_linear_probing.py` and `eegnet_cross-subject.py`, you should first execute the script  `eegnet_cross-subject_lightning.py`. These two pipelines use the same pre-trained neural networks and the `eegnet_cross-subject_lightning.py` script will actually train these networks and save them for later use.

By default, the configuration file is set so that the FBCSP and EEGNet+LP pipelines are evaluated by 5-folds cross-validation on all the calibration trials available. The results with random permutation cross-validation that use restricted number of calibration trials can be computed by editing the configuration files:
1. In `config.yaml`, comment the lines: 
```
n_perms: null
data_size: null
```
2. In `config.yaml`, uncomment the lines:
```
#    data_size:
#      policy: per_class
#      value: [1, 2, 4, 8, 16, 32, 64, 96]
#    n_perms: [50, 34, 23, 15, 10, 7, 5, 5]
```
3. In `local_config.yaml`, change the `suffix` field so that the previous results are not overwritten.


## Visualisation of the experiments
The plots and figures present in the article can be reproduced using the jupyter notebook `visualization.ipynb`. 

## Citing
To cite our work in a scientific publication, please use:
```
@article{
    TODO
}
```