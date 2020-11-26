import sys
sys.path.append('../..')

import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets, transforms
#from cifar10_loader import *
from rnn_loader import RnnCIFAR10, BlCIFAR10, DisRnnCIFAR10
import pandas as pd
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
#from func.tools.logger import Logger
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score, confusion_matrix

import crnn
from sklearn import metrics
from data_submit import get_csv_list

from torch.nn.utils.rnn import pack_padded_sequence

def seed_everything(seed, cuda=True):
  # Set the random seed manually for reproducibility.
  np.random.seed(seed)
  torch.manual_seed(seed)
  if cuda:
    torch.cuda.manual_seed_all(seed)

class Net(nn.Module):              # Define a class called Net, which makes it easy to call later.
    def __init__(self, in_channel, num_classes):           #Initialize the value of the instance. These values are generally used by other methods.
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 10, kernel_size=5)   # 2D convolutional layer
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # # 2D convolutional layer
        self.conv2_drop = nn.Dropout2d()           # Dropout layer
        self.fc1 = nn.Linear(500, 40)             #  change 50 to 10, for version2 test 
        self.fc2 = nn.Linear(40, num_classes)              # 

    def forward(self, x):                    # define the network using the module in __init__
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class ConvRNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, time_len):
        super(ConvRNN, self).__init__()
        #self.in_channels = in_channels
        #self.out_channels = out_channels
        self.time_len = time_len
        self.lstmcell = crnn.Conv2dLSTMCell(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(out_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # # 2D convolutional layer
        self.conv2_drop = nn.Dropout2d()           # Dropout layer
        self.fc1 = nn.Linear(500, 40)             # fully connected layer   
        self.fc2 = nn.Linear(40, num_classes)  
    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(self.time_len):
            if i == 0:
                hx, cx = self.lstmcell(x[i])
            else:
                hx, cx = self.lstmcell(x[i], (hx, cx))
        x = F.relu(F.max_pool2d(self.conv1(hx), 2))
        
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        x = x.view(-1, 500)
      
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ConvDisRNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, time_len):
        super(ConvDisRNN, self).__init__()
        #self.in_channels = in_channels
        #self.out_channels = out_channels
        #self.lstmcell = crnn.Conv2dLSTMCell(in_channels, out_channels, kernel_size)
        self.dislstmcell = crnn.LSTMdistCell('infor_exp', in_channels, out_channels, kernel_size, convndim = 2)
        #self.lstmcell = crnn.Conv2dLSTMCell(in_channels, out_channels, kernel_size)
        self.time_len = time_len
        self.conv1 = nn.Conv2d(out_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # # 2D convolutional layer
        self.conv2_drop = nn.Dropout2d()           # Dropout layer
        self.fc1 = nn.Linear(500, 40)             # fully connected layer
        self.fc2 = nn.Linear(40, num_classes)  
        
    def forward(self, x, time_dis):
        #print ('In network', x.shape)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #time_dis = torch.tensor(cfig['time_dis'])
        #print (time_dis)
        #print ('---------------', time_dis.shape)
        for i in range(self.time_len):
            
            if i == 0:
                hx, cx = self.dislstmcell(x[i], [time_dis[:,0], time_dis[:, 0]])
            else:
                hx, cx = self.dislstmcell(x[i], [time_dis[:,i-1], time_dis[:,i]], (hx, cx))  
                
        x = F.relu(F.max_pool2d(self.conv1(hx), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print ('In network', x.shape)
        x = x.view(-1, 500)
        #print ('In network', x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  
    
class ConvTRNN(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, time_len):
        super(ConvTRNN, self).__init__()
        #self.in_channels = in_channels
        #self.out_channels = out_channels
        #self.lstmcell = crnn.Conv2dLSTMCell(in_channels, out_channels, kernel_size)
        self.dislstmcell = crnn.TLSTMCell('TLSTMv2', in_channels, out_channels, kernel_size, convndim = 2)
        #self.lstmcell = crnn.Conv2dLSTMCell(in_channels, out_channels, kernel_size)
        self.time_len = time_len
        self.conv1 = nn.Conv2d(out_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # # 2D convolutional layer
        self.conv2_drop = nn.Dropout2d()           # Dropout layer
        self.fc1 = nn.Linear(500, 40)             # fully connected layer
        self.fc2 = nn.Linear(40, num_classes)  
        
    def forward(self, x, time_dis):
        #print ('In network', x.shape)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #time_dis = torch.tensor(cfig['time_dis'])
        #print (time_dis)
        #print ('---------------', time_dis.shape)
        for i in range(self.time_len):
            
            if i == 0:
                hx, cx = self.dislstmcell(x[i], [time_dis[:,0], time_dis[:, 0]])
            else:
                hx, cx = self.dislstmcell(x[i], [time_dis[:,i-1], time_dis[:,i]], (hx, cx))  # actually it's not very reasonable here, since the input gate and the forgate gate should have a different number.
                
        x = F.relu(F.max_pool2d(self.conv1(hx), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print ('In network', x.shape)
        x = x.view(-1, 500)
        #print ('In network', x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  
    
class DenseDisRNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_classes, time):
        super(DenseDisRNN, self).__init__()
        #self.in_channels = in_channels
        #self.out_channels = out_channels
        self.lstmcell = crnn.Conv2dLSTMCell(out_channels, out_channels, kernel_size)
        self.dislstmcell = crnn.LSTMdistCell(cfig['mode'], in_channels, out_channels, kernel_size, convndim = 2)
        #self.lstmcell = crnn.Conv2dLSTMCell(in_channels, out_channels, kernel_size)
        self.time = time
        self.conv1 = nn.Conv2d(2* out_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # # 2D convolutional layer
        self.conv2_drop = nn.Dropout2d()           # Dropout layer
        self.fc1 = nn.Linear(500, 10)             # fully connected layer
        self.fc2 = nn.Linear(10, num_classes)  
        
    def forward(self, x):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        time_dis = torch.tensor([0.8,0.3,0.]).cuda()  # [0.8, 0.5, 0]
        for i in range(self.time):
            if i == 0:
                hx1, cx1 = self.dislstmcell(x[i], time_dis[i])
            else:
                hx1, cx1 = self.dislstmcell(x[i], time_dis[i], (hx1, cx1))
                
        for i in range(1, self.time):      # Dense Connected, haven't tested 0207
            if i == 1:
                hx2, cx2 = self.dislstmcell(x[i], time_dis[i])
            else:
                hx2, cx2 = self.dislstmcell(x[i], time_dis[i], (hx2, cx2))
                
        for i in range(0, 3, 2):
            if i == 0:
                hx3, cx3 = self.dislstmcell(x[i], time_dis[i])
            else:
                hx3, cx3 = self.dislstmcell(x[i], time_dis[i], (hx3, cx3))
                
#         hx, cx = self.lstmcell(hx3)
#         hx, cx = self.lstmcell(hx2, (hx, cx))
#         hx, cx = self.lstmcell(hx1, (hx, cx))
            
        hx = torch.cat([hx1, hx3], 1)
        
        #x = (hx1 + hx2) / 2.0
        
        x = F.relu(F.max_pool2d(self.conv1(hx), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print ('In network', x.shape)
        x = x.view(-1, 500)
        #print ('In network', x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)     
        

class Trainer(object):

    def __init__(self, cfig):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfig = cfig
        self.csv_path = os.path.join(self.cfig['save_path'], 'csv')
        self.lr = cfig['learning_rate']
        
        if self.cfig['model_name'] == 'rnn':
            print ('====== rnn========')
            self.model = ConvRNN(in_channels = 3, out_channels = 3 * self.cfig['time'], kernel_size = 3, num_classes = self.cfig['n_classes'],time_len = self.cfig['time'] - self.cfig['begin_time']).to(self.device)    
        
        if self.cfig['model_name'] in ['disrnn']:
            print ('======distance rnn========')
            self.model = ConvDisRNN(in_channels = 3, out_channels = 3 * (self.cfig['time'] - self.cfig['begin_time']), kernel_size = 3, num_classes = self.cfig['n_classes'],time_len = self.cfig['time'] - self.cfig['begin_time']).to(self.device)  
       
        if self.cfig['model_name'] == 'dendisrnn':
            print ('====== Dense connected distance rnn========')
            self.model = DenseDisRNN(in_channels = 3, out_channels = 2* self.cfig['time'], kernel_size = 3, num_classes = self.cfig['n_classes'],time = self.cfig['time']).to(self.device) 
            
        if self.cfig['model_name'] == 'bl':
            print ('====== baseline ========')
            self.model = Net(in_channel = 3, num_classes = self.cfig['n_classes']).to(self.device)
        if self.cfig['model_name'] == 'trnn':
            print ('====== T LSTM ========')
            self.model = ConvTRNN(in_channels = 3, out_channels = 3 * (self.cfig['time'] - self.cfig['begin_time']), kernel_size = 3, num_classes = self.cfig['n_classes'],time_len = self.cfig['time'] - self.cfig['begin_time']).to(self.device) 
        
        img_label_dict = {}
        img_dis_dict = {}
        train_list,val_list, test_list = [], [], []
        df_tr = pd.read_csv(self.cfig['tr_csv_path'])
        for i , item in df_tr.iterrows():
            tmp_time_dis = get_csv_list(item['time'])
            tmp_time_dis = [max(tmp_time_dis) - i for i in tmp_time_dis]
            img_dis_dict[item['img']] = tmp_time_dis
            img_label_dict[item['img']] = int(item['gt'])
            if i % 5 == 0:
                val_list.append(item['img'])
            else:
                train_list.append(item['img'])
                
        df_tt = pd.read_csv(self.cfig['tt_csv_path'])
        for i , item in df_tt.iterrows():
            tmp_time_dis = get_csv_list(item['time'])
            tmp_time_dis = [max(tmp_time_dis) - i for i in tmp_time_dis]
            img_dis_dict[item['img']] = tmp_time_dis
            img_label_dict[item['img']] = int(item['gt'])
            test_list.append(item['img'])
                
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        #print (img_dis_dict)
        if self.cfig['model_name'] in ['disrnn', 'trnn']:
            print ('use distanced rnn loader')
            trainset = DisRnnCIFAR10(self.cfig['data_root'], train_list, img_label_dict, phase = 'train',  begin_time = self.cfig['begin_time'], time = self.cfig['time'], img_dis_dict = img_dis_dict, transform=train_transform)
            valset = DisRnnCIFAR10(self.cfig['data_root'], val_list, img_label_dict, phase = 'train',  begin_time = self.cfig['begin_time'], time = self.cfig['time'], img_dis_dict = img_dis_dict, transform=valid_transform)
            testset = DisRnnCIFAR10(self.cfig['data_root'], test_list, img_label_dict,phase = 'test',  begin_time = self.cfig['begin_time'], time = self.cfig['time'], img_dis_dict = img_dis_dict, transform=valid_transform)
        
        if self.cfig['model_name'] == 'rnn':    # the phase of valset is 'train' because image is in the train folder
            print ('use rnn loader')
            trainset = RnnCIFAR10(self.cfig['data_root'], train_list, img_label_dict, phase = 'train', begin_time = self.cfig['begin_time'], time = self.cfig['time'], transform=train_transform)
            valset = RnnCIFAR10(self.cfig['data_root'], val_list, img_label_dict, phase = 'train', begin_time = self.cfig['begin_time'], time = self.cfig['time'], transform=valid_transform)
            testset = RnnCIFAR10(self.cfig['data_root'], test_list, img_label_dict,phase = 'test', begin_time = self.cfig['begin_time'], time = self.cfig['time'], transform=valid_transform)
        if self.cfig['model_name'] == 'bl':
            print ('use bl loader')
            trainset = BlCIFAR10(self.cfig['data_root'], train_list, img_label_dict, phase = 'train',transform=train_transform)
            valset = BlCIFAR10(self.cfig['data_root'], val_list, img_label_dict, phase = 'train',transform=valid_transform)
            testset = BlCIFAR10(self.cfig['data_root'], test_list, img_label_dict, phase = 'test',transform=valid_transform)
        
        self.train_loader = torch.utils.data.DataLoader(trainset, 
                    batch_size=self.cfig['batch_size'], 
                    num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(valset, 
                    batch_size=self.cfig['batch_size'], 
                    num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(testset, 
                    batch_size=self.cfig['batch_size'], 
                    num_workers=4)
        
        
        print ('len train_loader: ', len(self.train_loader) )
        print ('len val_loader: ', len(self.val_loader) )
        print ('len test_loader: ', len(self.test_loader) )
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.lr, betas=(0.9, 0.999))
        
#        self.logger = Logger(osp.join(self.cfig['save_path'], 'logs'))
    
    def adjust_learning_rate(self, optimizer, epoch, steps, gamma):
        if epoch in steps:
            self.lr *= gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr   
                print ('After modify, the learning rate is', param_group['lr'])
        
    def train(self):
        for epoch in tqdm(range(self.cfig['start_epoch'], self.cfig['max_epoch'])):
            if self.cfig['adjust_lr']:
                self.adjust_learning_rate(self.optim, epoch, self.cfig['steps'], self.cfig['lr_gamma'])
                self.optim = torch.optim.Adam(self.model.parameters(), lr = self.lr, betas=(0.9, 0.999))
            model_root = osp.join(self.cfig['save_path'], 'models')
            if not os.path.exists(model_root):
                os.mkdir(model_root)
            model_pth = '%s/model_epoch_%04d.pth' % (model_root, epoch)
            if os.path.exists(model_pth) and self.cfig['use_exist_model']:
                if self.device == 'cuda': #there is a GPU device
                    self.model.load_state_dict(torch.load(model_pth))
                else:
                    self.model.load_state_dict(torch.load(model_pth, map_location=lambda storage, location: storage))
            else:
                self.train_epoch(epoch)
                if self.cfig['savemodel']:
                    torch.save(self.model.state_dict(), model_pth)
            if self.cfig['iseval']:
                self.eval_epoch(epoch)
                self.test_epoch(epoch)
    
    def train_epoch(self, epoch):
        self.model.train()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        train_csv = os.path.join(self.csv_path, 'train.csv')
        pred_list, target_list, loss_list, pos_list = [],[],[],[]
        print ('epoch: ', epoch)
        #print (self.model.dislstmcell.a)
        
        
        for batch_idx, item in enumerate(self.train_loader):
            if self.cfig['model_name'] in ['disrnn', 'trnn']:
                data, target, dist = item
                data, target, dist = data.to(self.device), target.to(self.device), dist.to(self.device)
            else:
                data, target, ID = item
                data, target = data.to(self.device), target.to(self.device)
            

            if self.cfig['model_name'][-3:] == 'rnn':
                data = data.permute([1,0,2,3,4])
            #print ('data shape', data.shape, self.cfig['batch_size'])
            #data = pack_padded_sequence(data, [3] * self.cfig['batch_size'])   # if use cell, we don't need it.
            self.optim.zero_grad()
            #print ('=================',data.shape)
            if self.cfig['model_name'] in ['disrnn', 'trnn']:
                pred = self.model(data, dist)
            else:
                pred = self.model(data)             # here should be careful
            pred_prob = F.softmax(pred)
            #loss = self.criterion(pred, target)
            #print (pred.shape, target.shape)
            if batch_idx == 0:
                print ('data.shape',data.shape)
                print ('pred.shape', pred.shape)
                print('Epoch: ', epoch)
            loss = nn.CrossEntropyLoss()(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)
            self.optim.step()
            print_str = 'train epoch=%d, batch_idx=%d/%d, loss=%.4f\n' % (
            epoch, batch_idx, len(self.train_loader), loss.data[0])
            #print(print_str)
            pred_cls = pred.data.max(1)[1]
            pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())
        try: 
            print (1000 * self.model.dislstmcell.a.grad, ' a grad')
            
            print (self.model.dislstmcell.a.data, self.model.dislstmcell.c.data)
            print (1000 * self.model.dislstmcell.c.grad, 'c grad')
            #print (self.model.dislstmcell.weight_ih.grad.max(), 'weight_ih')
        except:
            print ('a.grad none')    
        print (confusion_matrix(target_list, pred_list))
        accuracy=accuracy_score(target_list,pred_list)
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr)
        #-------------------------save to csv -----------------------#
        if not os.path.exists(train_csv):
            csv_info = ['epoch', 'loss', 'auc', 'accuracy']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(train_csv)
        df = pd.read_csv(train_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)
        tmp_auc = df['auc'].tolist()
        tmp_auc.append(roc_auc)
        #print('------------------', tmp_epoch)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        
        data['epoch'], data['loss'],data['auc'], data['accuracy'] =tmp_epoch, tmp_loss,tmp_auc, tmp_acc
        print ('train accuracy: ', accuracy, 'train auc: ', roc_auc)
        data.to_csv(train_csv)
        
        #---------------------- save to tensorboard ----------------#
#         if self.cfig['save_tensorlog']:
#             self.logger.scalar_summary('loss', np.mean(loss_list), epoch + 1)
#             self.logger.scalar_summary('accuracy', accuracy, epoch + 1)

        
    def eval_epoch(self, epoch):  
        #model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.save_path, 'models'), epoch)
        #self.model.load_state_dict(torch.load(model_pth))        # do these two impactful? I don't know
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, 'eval.csv')
        pred_list, target_list, loss_list, pos_list = [],[],[],[]
        for batch_idx, item in enumerate(self.val_loader):
            if self.cfig['model_name'] in ['disrnn', 'trnn']:
                data, target, dist = item
                data, target, dist = data.to(self.device), target.to(self.device), dist.to(self.device)
                if batch_idx == 0: print (dist.shape)
            else:
                data, target, ID = item
                data, target = data.to(self.device), target.to(self.device)
            
            if self.cfig['model_name'][-3:] == 'rnn':
                data = data.permute([1,0,2,3,4])
            #data = pack_padded_sequence(data, [3] * self.cfig['batch_size'])   # if use cell, we don't need this
            self.optim.zero_grad()
            if self.cfig['model_name'] in ['disrnn', 'trnn']:
                pred = self.model(data, dist)
            else:
                pred = self.model(data)   
            pred_prob = F.softmax(pred)
            #loss = self.criterion(pred, target)
            loss = nn.CrossEntropyLoss()(pred, target)
            
            pred_cls = pred.data.max(1)[1]  # not test yet
            pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())
            
        
        accuracy=accuracy_score(target_list,pred_list)
        print (confusion_matrix(target_list, pred_list))
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr) 
        #-------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'loss', 'auc', 'accuracy']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(eval_csv)
        df = pd.read_csv(eval_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)

        #print ('------------------', tmp_epoch)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_auc = df['auc'].tolist()
        tmp_auc.append(roc_auc)
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        
        data['epoch'], data['loss'],data['auc'], data['accuracy'] =tmp_epoch, tmp_loss,tmp_auc, tmp_acc
        data.to_csv(eval_csv)
        print ('val accuracy: ', accuracy  , 'val auc: ', roc_auc)
        print ('max val auc at: ', max(tmp_auc), tmp_auc.index(max(tmp_auc)))
        
        #---------------------- save to tensorboard ----------------#
#         if self.cfig['save_tensorlog']:
#             self.logger.scalar_summary('val_loss', np.mean(loss_list), epoch + 1)
            
#             self.logger.scalar_summary('val_accuracy', accuracy, epoch + 1)

        
    def test_epoch(self, epoch):
        self.model.eval()
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        eval_csv = os.path.join(self.csv_path, 'test.csv')
        pred_list, target_list, loss_list, pos_list = [],[],[], []
        for batch_idx, item in enumerate(self.test_loader):
            if self.cfig['model_name'] in ['disrnn', 'trnn']:
                data, target, dist = item
                data, target, dist = data.to(self.device), target.to(self.device), dist.to(self.device)
            else:
                data, target, ID = item
                data, target = data.to(self.device), target.to(self.device)
            
            if self.cfig['model_name'][-3:] == 'rnn':
                data = data.permute([1,0,2,3,4])
            #data = pack_padded_sequence(data, [3] * self.cfig['batch_size'])   # if use cell, we don't need this
            self.optim.zero_grad()
            if self.cfig['model_name'] in ['disrnn', 'trnn']:
                pred = self.model(data, dist)
            else:
                pred = self.model(data)   
            pred_prob = F.softmax(pred)
            #loss = self.criterion(pred, target)
            loss = nn.CrossEntropyLoss()(pred, target)
            
            pred_cls = pred.data.max(1)[1]  # not test yet
            pos_list += pred_prob[:, 1].data.cpu().numpy().tolist()
            pred_list += pred_cls.data.cpu().numpy().tolist()
            target_list += target.data.cpu().numpy().tolist()
            loss_list.append(loss.data.cpu().numpy().tolist())
            
        
        accuracy=accuracy_score(target_list,pred_list)
        print (confusion_matrix(target_list, pred_list))
        fpr, tpr, threshold = metrics.roc_curve(target_list, pos_list)
        roc_auc = metrics.auc(fpr, tpr) 
        #-------------------------save to csv -----------------------#
        if not os.path.exists(eval_csv):
            csv_info = ['epoch', 'loss', 'auc','accuracy']
            init_csv = pd.DataFrame()
            for key in csv_info:
                init_csv[key] = []
            init_csv.to_csv(eval_csv)
        df = pd.read_csv(eval_csv)
        data = pd.DataFrame()
        tmp_epoch = df['epoch'].tolist()
        tmp_epoch.append(epoch)
        tmp_auc = df['auc'].tolist()
        tmp_auc.append(roc_auc)
        #print ('------------------', tmp_epoch)
        tmp_loss = df['loss'].tolist()
        tmp_loss.append(np.mean(loss_list))
        tmp_acc = df['accuracy'].tolist()
        tmp_acc.append(accuracy)
        
        data['epoch'], data['loss'],data['auc'], data['accuracy'] =tmp_epoch, tmp_loss,tmp_auc, tmp_acc
        data.to_csv(eval_csv)
        print ('test accuracy: ', accuracy, 'test auc: ', roc_auc)
        
       
    
    def test(self):
        model_root = osp.join(self.save_path, 'models')
        model_list = os.listdir(model_root)
        Acc, F1, Recl, Prcn = [], [], [], []
        for epoch in range(len(model_list)):
            print ('epoch: ', epoch)
            model_pth = '%s/model_epoch_%04d.pth' % (osp.join(self.save_path, 'models'), epoch)
            accuracy, f1, recall, precision = self.test_epoch(model_pth)
            print (accuracy, f1, recall, precision)
            Acc.append(accuracy)
            F1.append(f1)
            Recl.append(recall)
            Prcn.append(precision)
        data = pd.DataFrame()
        data['accuracy'] = Acc
        data['f1'] = F1
        data['recall'] = Recl
        data['precision'] = Prcn
        print ('Acc: ', Acc)
        print ('f1:', F1)
        print ('Recl', Recl)
        print ('Prcn', Prcn)
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)
        test_csv = os.path.join(self.csv_path, 'test.csv')
        data.to_csv(test_csv)
        
import yaml
import shutil        
if __name__ == '__main__':
    f = open('cifar10.yaml', 'r').read()
    cfig = yaml.load(f)
    shutil.copyfile('./cifar10.yaml', cfig['save_path'] + '/tmp.yaml')
    seed_everything(seed=1337, cuda=True)
    trainer = Trainer(cfig)
    trainer.train()