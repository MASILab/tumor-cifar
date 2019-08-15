# tumor-cifar

This is the supplementary material for the paper of MICCAIW-MLMI 2019: 

"Distanced LSTM: Time-Distanced Gates in LSTM to adapt Longitudinal Lung Cancer Detection". 
 

---------------------------------------------------------------------------------------------

In the python code script we have 4 functions:

1. get_csv_v1:

This function generate the meta information for tumor-CIFAR-v1 and save it in a csv file. The meta information includes: image name, image time point, nodule position, ground truth (cancer or non-cancer) and nodule size

2. get_csv_v2:

This function is for tumor-CIFAR-v2, serve the same function as get_csv_v1.

3. get_nodule_img:

Generating the image according to meta information from csv file.

4. add_nodule:

This function is called by get_nodule_img, which transfer the nodule information to image and add noise.


--------------------------------------

There is a example to show how to use the data in demo_submit.ipynb
