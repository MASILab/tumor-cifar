import numpy as np 
from skimage import io
import pandas as pd
import re
import os
import random

def get_csv_list(str_list):
    l = re.split('[][, ]', str_list)
    try:
        return [int(i) for i in l if len(i) != 0]
    except:
        return [float(i) for i in l if len(i) != 0]

def get_csv_v2(data_root, save_path):
    '''
    data_root: the original CIFAR10 image root
    save_path: the save path of generated csv file 
    '''
    img_list = os.listdir(data_root)
    T = []
    nodule_list = []
    Gt = []
    Size = []

    for i in range(len(img_list)):

        tmp_interval = np.random.normal(5, 3, 5)
        tmp_interval = [abs(t) for t in tmp_interval]
        
        tmp_adT = np.zeros(5)
        for i in range(1, 5):
            tmp_adT[i] = sum(tmp_interval[: i])
        
        tmp_growrate = abs(np.random.normal(1, 0.2))  # growrate only one number
        
        tmp_size = tmp_adT * tmp_growrate 
        
        gt = int(np.random.uniform(0,1) < 0.5)
        
        if gt == 1:
            tmp_adT = tmp_adT / 3.0

        nodule_list.append(list(np.random.randint(7, 25, 4)))  # 10 because there are 5 images each sample
        Gt.append(gt)

        tmp_adT = np.around(tmp_adT, decimals = 3)
        tmp_size = np.around(tmp_size, decimals = 3)
        
        T.append(tmp_adT)

        Size.append(tmp_size)
        
    data = pd.DataFrame()
    data['img'] = img_list
    data['time'] = T
    data['nodules'] = nodule_list
    data['gt'] = Gt
    data['size'] = Size
    data.to_csv(save_path, index=False)
    
def get_csv_v1(data_root, save_path):
    '''
    data_root: the original CIFAR10 image root
    save_path: the save path of generated csv file 
    '''
    img_list = os.listdir(data_root)
    T = []
    nodule_list = []
    Gt = []
    Size = []

    for i in range(len(img_list)):

        tmp_interval = np.random.normal(1.67, 1, 5)
        tmp_interval = [abs(t) for t in tmp_interval]
        
        tmp_adT = np.zeros(5)
        for i in range(1, 5):
            tmp_adT[i] = sum(tmp_interval[: i])
        
        tmp_growrate = abs(np.random.normal(1, 0.2))  # growrate only one number
        
        
        
        gt = int(np.random.uniform(0,1) < 0.5)
        
        if gt == 1:
            tmp_growrate = tmp_growrate * 3.0

        tmp_size = tmp_adT * tmp_growrate 
        nodule_list.append(list(np.random.randint(7, 25, 4)))  
        Gt.append(gt)

        tmp_adT = np.around(tmp_adT, decimals = 3)
        tmp_size = np.around(tmp_size, decimals = 3)
        
        T.append(tmp_adT)

        Size.append(tmp_size)
        
    data = pd.DataFrame()
    data['img'] = img_list
    data['time'] = T
    data['nodules'] = nodule_list
    data['gt'] = Gt
    data['size'] = Size
    data.to_csv(save_path, index=False)
    
    
def get_nodule_img(data_root, csv_path, save_root, sp_prob):   
    df = pd.read_csv(csv_path)
    for t in range(5):
        if not os.path.exists(save_root + '/T' + str(t)):
            os.makedirs(save_root + '/T'  + str(t))
    for i, item in df.iterrows():
        
        im_path = os.path.join(data_root, item['img'])
        nodules = get_csv_list(item['nodules'])
        rs = get_csv_list(item['size'])
        
        if i %1000 == 1: print (i, ' have been finished!') 
        imgs = add_nodule(im_path, nodules, rs, sp_prob)
        
        assert len(imgs) == 5
        io.imsave(save_root + '/T0/' + item['img'], imgs[0])
        io.imsave(save_root + '/T1/' + item['img'], imgs[1])
        io.imsave(save_root + '/T2/' + item['img'], imgs[2])
        io.imsave(save_root + '/T3/' + item['img'], imgs[3])
        io.imsave(save_root + '/T4/' + item['img'], imgs[4])
 
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = [20,20,20]
            elif rdn > thres:
                output[i][j] = [220, 220, 220]
            else:
                output[i][j] = image[i][j]
    return output

def add_nodule(im_path, nodules, rs, sp_prob):  
    imgs = np.zeros((5, 32, 32, 3), dtype = np.uint8)
    rand = np.random.uniform(0,1)
    for t in range(5):
        img = io.imread(im_path)
        
        # ---- if add possion noise -------# 
#         noisy = np.random.poisson(img.astype(float)).astype(float)
#         noisy = 10 * ((noisy - np.min(noisy)) / (np.max(noisy) - np.min(noisy)))
#         noisy = noisy.astype('uint8')
#         img = img + noisy
        # ---------------------------------#
        
        for i in range(nodules[0] - 5, nodules[0]+ 6):
            for j in range(nodules[1] - 5, nodules[1]+6):
                if (i - nodules[0]) * (i - nodules[0]) + (j - nodules[1]) * (j - nodules[1]) < 0.3 * rs[t]:
                    img[i, j, :] = np.minimum(250, (1 + 0.1 * rs[t]) * img[i, j, :])
                    
        for i in range(nodules[2] - 5, nodules[2]+6):
            for j in range(nodules[3] - 5, nodules[3]+6):
                if (i - nodules[2]) * (i - nodules[2]) + (j - nodules[3]) * (j - nodules[3]) < 0.3 * rs[t]:
                    img[i, j, :] = np.minimum(250, (1 + 0.1 * rs[t]) * img[i, j, :])
        
        # --- if add salt and pepper noise #
        img = sp_noise(img,sp_prob)
        
        imgs[t] = img
    return imgs  
