from __future__ import print_function

import argparse
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as tr

from tqdm import tqdm
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score

from torch.utils.data.dataset import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as tr

class CNNModel(nn.Module):
    def __init__(self, in_channels, num_classes, is_binary):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(in_channels, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(2**3*64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)     
        self.sigmoid = nn.Sigmoid()   
        self.is_binary = is_binary
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)

        if self.is_binary:
          out = self.sigmoid(out)
        
        return out

class SimpleModel(nn.Module):
   def __init__(self, set_size):
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(set_size, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, 1),
            nn.Sigmoid()
        ) 
    
   def forward(self, input):
        x = input
        x = self.model(x)
        return x

class DeepSetSimple(nn.Module):
    def __init__(self, age_range, in_features, set_features=50):
        super(DeepSetSimple, self).__init__()

        self.embedding = nn.Embedding(age_range, 100)
        self.fc = nn.Linear(100, 30)
        self.fc1 = nn.Linear(30,1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = input
        x = self.embedding(x)
        x = self.fc(x)
        x = self.tanh(x)
        x = x.sum(dim=1)
        x = self.fc1(x)
        return self.sigmoid(x)


class DeepSet(nn.Module):

    def __init__(self, set_size, in_features, set_features=100): #50
        super(DeepSet, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 200), #100
            nn.ELU(inplace=True),
            nn.Linear(200, 100), #100
            nn.ELU(inplace=True),
            nn.Linear(100, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 30),
            nn.ELU(inplace=True),
            nn.Linear(30, 10),
            nn.ELU(inplace=True),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

        self.weighted_sum = nn.Conv1d(in_channels=set_size, out_channels=1, kernel_size=1)
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, input):
        x = input
        x = self.feature_extractor(x)
        #x = self.weighted_sum(x)
        x = x.sum(dim=1) 
        x = self.regressor(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'

class DatasetIncome(Dataset):
  def __init__(self, n, X, y):

    self.n = n

    #load files
    self.X = X
    self.y = y

  def __getitem__(self, index):
    
    item = self.X[index,:]
    item = item.transpose((3,0,1,2)) 
    return torch.from_numpy(item).type(torch.FloatTensor), torch.from_numpy(np.array([self.y[index]])).type(torch.FloatTensor)

  def __len__(self):
    return self.n

class DatasetIncomeDeepset(Dataset):
  def __init__(self, n, X, y):

    self.n = n

    #load files
    self.X = X
    self.y = y

  def __getitem__(self, index):
    
    item = self.X[index,:]

    # for i in range(item.shape[0]):
    #     if item[i,0] == 0.2876712328767123:
    #         item[i,13] = item[i,13] + 2.0

    item = item.reshape((item.shape[0],26))
    
    #item = item.reshape((item.shape[0],14)) # for baseline


    #item.sort()
    #item = item[:,:14]
    #item = item[item[:, 0].argsort()] # sort by age
    #return torch.tensor(item).to(torch.int64), torch.from_numpy(np.array([self.y[index]])).type(torch.FloatTensor)

    return torch.from_numpy(item).type(torch.FloatTensor), torch.from_numpy(np.array([self.y[index]])).type(torch.FloatTensor)

  def __len__(self):
    return self.n

parser = argparse.ArgumentParser(description='DeepSets')
parser.add_argument('-p', '--epochs', metavar='E', type=int, default=350,
                  help='Number of epochs', dest='epochs')
parser.add_argument('-a', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                  help='Batch size', dest='batchsize')
parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                  help='Learning rate', dest='lr')
parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                  help='Load model from a .pth file')
parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                  help='Downscaling factor of the images')
parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                  help='Percent of the data that is used as validation (0-100)')
parser.add_argument('-d', '--dataset', dest='dataset', type=str, default=False,
                  help='Where to save generated data')
parser.add_argument('-b', '--bug', dest='bug_type', type=str, default=False,
                  help='What kind of bug')
parser.add_argument('-e', '--exp', dest='exp_type', type=str, default=False,
                  help='What kind of explanation')
parser.add_argument('-x', '--set-size', metavar='B', type=int, nargs='?', default=32,
                  help='Batch size', dest='set_size')
parser = parser.parse_args()


#args.cuda = args.cuda and torch.cuda.is_available()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device", device)

X = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"_1200_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
y = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"_1200_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

# X = np.load('/content/drive/MyDrive/wifi_dataset/dataset_shap_450_labelleak.npy')
# y = np.load('/content/drive/MyDrive/wifi_dataset/bugs_shap_450_labelleak.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

dset_train = DatasetIncomeDeepset(X_train.shape[0], X_train, y_train)

train_loader = DataLoader(dset_train,
                          batch_size=parser.batchsize,
                          shuffle=True, num_workers=1)
dset_test = DatasetIncomeDeepset(X_test.shape[0], X_test, y_test)

test_loader = DataLoader(dset_test,
                         batch_size=parser.batchsize,
                         shuffle=False, num_workers=1)


print("Training Data : ", len(train_loader.dataset))
print("Testing Data : ", len(test_loader.dataset))

# %% Loading in the model

if parser.exp_type == 'baseline':
  #model = CNNModel(1, 1, True)
  model = DeepSet(parser.set_size, 13+1)
else:
  #model = SimpleModel(parser.set_size)
  model = DeepSet(parser.set_size, 13+13)
  #model = DeepSetSimple(parser.set_size, 13+1)
  #model = CNNModel(3, 1, True)
model.to(device=device)

# if args.optimizer == 'SGD':
#     optimizer = optim.SGD(model.parameters(), lr=args.lr,
#                           momentum=0.99)
# if args.optimizer == 'ADAM':
#     optimizer = optim.Adam(model.parameters(), lr=args.lr,
#                            betas=(0.9, 0.999))

#optimizer = optim.Adam(model.parameters(), lr=parser.lr, betas=(0.9, 0.999))

optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-3)

# Defining Loss Function
#criterion = DICELoss()
criterion = nn.BCELoss()

#criterion = nn.L1Loss()

# Define Training Loop


def train(epoch, loss_list):
    model.train()
    for batch_idx, (image, mask) in enumerate(train_loader):

        image = image.to(device=device)
        mask = mask.to(device=device)
        mask = mask.squeeze(1)
        image, mask = Variable(image), Variable(mask)

        optimizer.zero_grad()

        output = model(image)

        loss = criterion(output, mask.unsqueeze(1)) 
        loss_list.append(loss.item())
        #print(loss.item())

        loss.backward()
        optimizer.step()

        # if args.clip:
        #     nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

        #if batch_idx % args.log_interval == 0:
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(image), len(train_loader.dataset),
        #     100. * batch_idx / len(train_loader), loss.item()))


def test(train_accuracy=False, save_output=False):
    model.eval()
    test_loss = 0
    correct = 0

    if train_accuracy:
        loader = train_loader
    else:
        loader = test_loader

    accs = []

    for batch_idx, (image, mask) in enumerate(loader):

        image = image.to(device=device)
        mask = mask.to(device=device)
        mask = mask.squeeze(1)
        image, mask = Variable(image), Variable(mask)

        optimizer.zero_grad()

        output = model(image)

        pred = np.round(output.detach())
        mask = np.round(mask.detach())   

        accs.append(accuracy_score(mask.tolist(),pred.tolist()))          

    if train_accuracy:
      print("Accuracy on train set is", sum(accs)/len(accs))
    else:
      print("Accuracy on test set is", sum(accs)/len(accs))

is_train = True

if is_train:
    loss_list = []
    for i in range(parser.epochs):
        train(i, loss_list)
        test(train_accuracy=True, save_output=False)
        test(train_accuracy=False, save_output=False)

else:
    model.load_state_dict(torch.load(args.load))
    test(save_output=True)
    test(train_accuracy=True)
