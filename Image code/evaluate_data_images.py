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
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

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

#code from https://github.com/yassersouri/pytorch-deep-sets/blob/5190e3eee8a438a0e6f599882786502d8fa0b09e/src/deepsets/networks.py#L33

class SmallCNNPhi(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5)
        self.conv4 = nn.Conv2d(40, 80, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(8000, 50)
        self.fc1_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 8000)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x) #why is this droupout2d??
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class SmallRho(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, self.output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x

class LargerCNNPhi(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        modules=list(resnet.children())[:-2]
        self.resnet=nn.Sequential(*modules)
        for p in resnet.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(128+128, 100)
        self.fc1 = nn.Linear(100, 1)
        self.fc2 = nn.Linear(10, 128)
        self.fc3 = nn.Linear(2048*7*7, 128)
        self.fc31 = nn.Linear(2048*7*7, 128)
        self.fc4 = nn.Linear(128, 10)
        self.sigmoid = nn.Sigmoid()
        self.drop=nn.Dropout(p=0.5) 

    def forward(self, img, saliency):
        img = self.resnet(img)
        img = img.reshape((-1, 2048*7*7))
        img = self.drop(img)
        img = F.relu(self.fc3(img))
        
        saliency = self.resnet(saliency)
        saliency = saliency.reshape((-1, 2048*7*7))
        saliency = self.drop(saliency)
        saliency = F.relu(self.fc31(saliency))

        comb = torch.cat((img, saliency), 1)
        x = F.relu(self.fc(comb))
        x = self.sigmoid(self.fc1(x))
        return x

class LargerCNNPhiBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        modules=list(resnet.children())[:-2]
        self.resnet=nn.Sequential(*modules)
        for p in resnet.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(128+128, 10)
        self.fc1 = nn.Linear(100, 1)
        self.fc2 = nn.Linear(10, 128)
        self.fc3 = nn.Linear(2048*7*7, 128)
        self.fc4 = nn.Linear(128, 10)
        self.sigmoid = nn.Sigmoid()
        self.drop=nn.Dropout(p=0.5) 

    def forward(self, img, saliency):
        img = self.resnet(img)
        img = img.reshape((-1, 2048*7*7))
        img = self.drop(img)
        img = F.relu(self.fc3(img))
        # img = self.drop(img)
        # img = F.relu(self.fc4(img))
        saliency = F.relu(self.fc2(saliency))
        comb = torch.cat((img, saliency), 1)
        x = F.relu(self.fc(comb))
        #x = self.sigmoid(self.fc1(x))
        return x

#code modified from https://github.com/yassersouri/pytorch-deep-sets/blob/5190e3eee8a438a0e6f599882786502d8fa0b09e/src/deepsets/networks.py
class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x, y):
        # compute the representation for each data point
        x = self.phi.forward(x, y)        
        return x

        #x = torch.sum(x, dim=0, keepdim=True) # right now set size is only 1.
        # compute the output
        #out = self.rho.forward(x)

        #return out

class DatasetIncomeBaseline(Dataset):
  def __init__(self, n, X, y):

    self.n = n

    #load files
    self.X = X
    self.y = y

  def __getitem__(self, index):
    
    item = self.X[index,:]
    item = item.reshape((224,224,4))
    item = item.transpose((2,0,1)) 
    return torch.from_numpy(item).type(torch.FloatTensor), torch.from_numpy(item).type(torch.FloatTensor), torch.from_numpy(np.array([self.y[index]])).type(torch.FloatTensor)

  def __len__(self):
    return self.n

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

    img = item[:,:,:, :3]
    saliency = item[:,:,:,-3:]

    img = img.reshape((224,224,3))
    saliency = saliency.reshape((224,224,3))

    img = img.transpose((2,0,1))
    saliency = saliency.transpose((2,0,1))

    #img = torch.from_numpy(img).type(torch.FloatTensor)
    #preprocess = transforms.Compose([
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #])

    return torch.from_numpy(img).type(torch.FloatTensor), torch.from_numpy(saliency).type(torch.FloatTensor), torch.from_numpy(np.array([self.y[index]])).type(torch.FloatTensor)

  def __len__(self):
    return self.n

class DatasetIncomeDeepsetBaseline(Dataset):
  def __init__(self, n, X, y):

    self.n = n

    #load files
    self.X = X
    self.y = y

  def __getitem__(self, index):
    
    item = self.X[index,:]

    saliency = np.zeros((5, 10,))
    for i in range(5):
        ind = item[i,0,0,3]
        ind = int(ind * 10)
        saliency[i][ind] = 1.

    temp_img = item[:,:,:, :3]
    img = np.zeros((5,3, 224, 224))
    for i in range(5):
        img[i] = temp_img[i,:,:,:].reshape((224,224,3)).transpose((2,0,1))

    img = torch.from_numpy(img).type(torch.FloatTensor)

    return img, torch.from_numpy(saliency).type(torch.FloatTensor), torch.from_numpy(np.array([self.y[index]])).type(torch.FloatTensor)

  def __len__(self):
    return self.n


# class DatasetIncomeDeepsetBaseline(Dataset):
#   def __init__(self, n, X, y):

#     self.n = n

#     #load files
#     self.X = X
#     self.y = y

#   def __getitem__(self, index):
    
#     item = self.X[index,:]

#     img = item[:,:,:, :3]
#     ind = item[0,0,0,3]
#     ind = int(ind * 10)
#     saliency = np.zeros((10,))
#     saliency[ind] = 1.

#     img = img.reshape((224,224,3))
#     img = img.transpose((2,0,1))
#     img = torch.from_numpy(img).type(torch.FloatTensor)
#     #preprocess = transforms.Compose([
#     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     #])
#     #img = preprocess(img)

#     return img, torch.from_numpy(saliency).type(torch.FloatTensor), torch.from_numpy(np.array([self.y[index]])).type(torch.FloatTensor)

#   def __len__(self):
#     return self.n




parser = argparse.ArgumentParser(description='DeepSets')
parser.add_argument('-p', '--epochs', metavar='E', type=int, default=350,
                  help='Number of epochs', dest='epochs')
parser.add_argument('-a', '--batch-size', metavar='B', type=int, nargs='?', default=32,
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

X = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
y = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

X1 = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"1_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
y1 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"1_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

X2 = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"2_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
y2 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"2_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

X3 = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"3_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
y3 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"3_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

# X4 = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"4_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
# y4 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"4_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

# X5 = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"5_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
# y5 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"5_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

# X6 = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"6_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
# y6 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"6_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

X_val = np.load("../all_datasets/"+parser.dataset+"_dataset/datasetval_"+parser.exp_type+"_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
y_val = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

# X_val1 = np.load("../all_datasets/"+parser.dataset+"_dataset/datasetval_"+parser.exp_type+"1_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
# y_val1 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"1_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

# X_val2 = np.load("../all_datasets/"+parser.dataset+"_dataset/datasetval_"+parser.exp_type+"2_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
# y_val2 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"2_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

X = np.concatenate((X,X1,X2,X3))
y = np.concatenate((y,y1,y2,y3))


# X = np.concatenate((X,X1,X2,X3,X4,X5,X6))
# y = np.concatenate((y,y1,y2,y3,y4,y5,y6))

# X_val = np.concatenate((X_val, X_val1, X_val2))
# y_val = np.concatenate((y_val, y_val1, y_val2))

#cluster into 5

# X_new = np.zeros((X.shape[0]//5,5,224,224,4))
# y_new = np.zeros((y.shape[0]//5))

# for i in range(X.shape[0]//5):
#     for j in range(5):
#         X_new[i][j] = X[int(i*5)+j]
#     y_new[i] = y[int(i*5)]


# X_val_new = np.zeros((X_val.shape[0]//5,5,224,224,4))
# y_val_new = np.zeros((y_val.shape[0]//5))

# for i in range(X_val.shape[0]//5):
#     for j in range(5):
#         X_val_new[i][j] = X_val[int(i*5)+j]
#     y_val_new[i] = y_val[int(i*5)]

# %% Loading in the model

if parser.exp_type == 'baseline':
  dset_train = DatasetIncomeDeepsetBaseline(X_new.shape[0], X_new, y_new)
  dset_test = DatasetIncomeDeepsetBaseline(X_val_new.shape[0], X_val_new, y_val_new)

  #model = LargerCNNPhiBaseline()

  # dset_train = DatasetIncomeBaseline(X.shape[0], X, y)
  # dset_test = DatasetIncomeBaseline(X_val.shape[0], X_val, y_val)
  model = InvariantModel(LargerCNNPhiBaseline(), SmallRho(input_size=10, output_size=1))
else:
  dset_train = DatasetIncomeDeepset(X.shape[0], X, y)
  dset_test = DatasetIncomeDeepset(X_val.shape[0], X_val, y_val)
  #model = LargerCNNPhi()
  model = InvariantModel(LargerCNNPhi(), SmallRho(input_size=10, output_size=1))
model.to(device=device)


train_loader = DataLoader(dset_train,
                          batch_size=parser.batchsize,
                          shuffle=True, num_workers=1)

test_loader = DataLoader(dset_test,
                         batch_size=parser.batchsize,
                         shuffle=True, num_workers=1)


print("Training Data : ", len(train_loader.dataset))
print("Testing Data : ", len(test_loader.dataset))


# if args.optimizer == 'SGD':
#     optimizer = optim.SGD(model.parameters(), lr=args.lr,
#                           momentum=0.99)
# if args.optimizer == 'ADAM':
#     optimizer = optim.Adam(model.parameters(), lr=args.lr,
#                            betas=(0.9, 0.999))

#optimizer = optim.Adam(model.parameters(), lr=parser.lr, betas=(0.9, 0.999))

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)

# Defining Loss Function
#criterion = DICELoss()
criterion = nn.BCELoss()

#criterion = nn.L1Loss()

# Define Training Loop


def train(epoch, loss_list):
    model.train()
    for batch_idx, (image, saliency, mask) in enumerate(train_loader):

        image = image.to(device=device)
        saliency = saliency.to(device=device)

        #image = image.reshape((5,3,224,224))
        #saliency = saliency.reshape((5,10))

        mask = mask.to(device=device)
        mask = mask.squeeze(1)
        image, saliency, mask = Variable(image), Variable(saliency), Variable(mask)

        optimizer.zero_grad()

        output = model(image, saliency)

        loss = criterion(output, mask.unsqueeze(1)) 
        loss_list.append(loss.item())

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

    for batch_idx, (image, saliency, mask) in enumerate(loader):

        image = image.to(device=device)
        saliency = saliency.to(device=device)

        #image = image.reshape((5,3,224,224))
        #saliency = saliency.reshape((5,10))

        mask = mask.to(device=device)
        mask = mask.squeeze(1)
        image, saliency, mask =  Variable(image), Variable(saliency), Variable(mask)

        optimizer.zero_grad()

        output = model(image, saliency)

        pred = np.round(output.cpu().detach())
        mask = np.round(mask.cpu().detach())   

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
