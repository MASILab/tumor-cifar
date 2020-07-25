# Tumor-CIFAR

This is the supplementary material for the oral-presentation paper of MICCAIW-MLMI 2019: 

"Distanced LSTM: Time-Distanced Gates in LSTM to adapt Longitudinal Lung Cancer Detection". 

and Journal Version: 

"Time-Distanced Gates in Long Short Term Memory Networks", Medical Image Analysis, 2020 (IF=11.15).

Please cite our paper if you find our work is helpful to you. 


---------------------------------------------------------------------------------------------

## Functions to Generate Simulation Set

In the python code script we have 4 functions:

1. get_csv_v1:

This function generate the meta information for tumor-CIFAR-v1 and save it in a csv file. The meta information includes: image name, image time point, nodule position, ground truth (cancer or non-cancer) and nodule size.

2. get_csv_v2:

This function is for tumor-CIFAR-v2, serve the same function as get_csv_v1.

3. get_nodule_img:

Generating the image according to meta information from csv file.

4. add_nodule:

This function is called by get_nodule_img, which transfer the nodule information to image and add noise.

<img src="https://github.com/MASILab/tumor-cifar/blob/master/figure.png" width="600">

--------------------------------------

## Usage

### Generate Dataset
There is an example showing how to use the data in demo_submit.ipynb.

### training

python new_main.py

Note there is a config file named cifar10.yaml

<img src="https://github.com/MASILab/tumor-cifar/blob/master/illustration.png" width="600">

--------------------------------------

## Document

The file TumorCIFAR_materials.pdf describes why and how we create the Tumor-CIFAR. Please email Riqiang Gao (riqiang.gao@vanderbilt.edu) if you have further concerns. 
