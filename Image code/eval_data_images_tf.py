import random
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from numpy import random as random1
import matplotlib.pyplot as plt 
import argparse
from load_datasets import *

from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D, SpatialDropout2D, Concatenate, Input
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import regularizers
import tensorflow as tf
import keras

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

#Data generator

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras."""
    def __init__(self, img_files, labels, exp_type=None, ave=None, std=None, batch_size=16, dim=(224, 224), n_channels=3,
                 n_classes=2, shuffle=True):
        """Initialization.
        
        Args:
            img_files: A list of path to image files.
            clinical_info: A dictionary of corresponding clinical variables.
            labels: A dictionary of corresponding labels.
        """
        self.img_files = img_files
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.exp_type = exp_type
        if ave is None:
            self.ave = np.zeros(n_channels)
        else:
            self.ave = ave
        if std is None:
            self.std = np.zeros(n_channels) + 1
        else:
            self.std = std
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.img_files) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.img_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples."""
        # X : (n_samples, *dim, n_channels)
        # X = [np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))]
        X_img = []
        X_clinical = []
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ind in enumerate(indexes):
            curr_item = self.img_files[ind]
            img= curr_item[:,:,:,:3]
            img = img.reshape((224,224,3))
            X_img.append(img)

            if self.exp_type == 'baseline':
                ind = curr_item[0,0,0,3]
                ind = int(ind * 10)
                saliency = np.zeros((10))
                saliency[ind] = 1.
            else:
                saliency= curr_item[:,:,:,-3:]
                saliency = saliency.reshape((224,224,3))

            X_clinical.append(saliency)
            y[i] = self.labels[ind]

        temp = np.array(X_img)
        temp1 = np.array(X_clinical)

        X = [temp, temp1]
        return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)

#train original model

if parser.bug_type == 'sc':
    train_it = load_debugging_tests_SC(False)
else:
    train_it = load_debugging_tests(False, parser.bug_type)

img_height,img_width = 224,224 
num_classes = 10
base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape= (img_height,img_width,3), )

for layer in base_model.layers[:]:
 layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dropout(0.5)(x)
final_output = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001), name="dense_b")(x)
x = Dropout(0.5)(final_output)
predictions = Dense(num_classes, activation= 'softmax', name="dense_a")(x)
model = Model(inputs = base_model.input, outputs = predictions)


#predictions1 = Dense(num_classes, activation= 'softmax')(x1)
#model1 = Model(inputs = base_model1.input, outputs = predictions1)

#TRAIN THE FIRST MODEL
#from keras.optimizers import SGD, Adam
#adam = Adam(lr=0.001)
#model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit_generator(train_it_original, epochs = 1, steps_per_epoch = 3346 // 16)

#load the new data

X = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
y = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

X1 = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"1_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
y1 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"1_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

X2 = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"2_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
y2 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"2_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

# X3 = np.load("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"3_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
# y3 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"3_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

X = np.concatenate((X,X1))
y = np.concatenate((y,y1))

X_val = X2
y_val = y2

#X_val = np.load("../all_datasets/"+parser.dataset+"_dataset/datasetval_"+parser.exp_type+"_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
#y_val = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

# X_val1 = np.load("../all_datasets/"+parser.dataset+"_dataset/datasetval_"+parser.exp_type+"1_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
# y_val1 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"1_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

# X_val2 = np.load("../all_datasets/"+parser.dataset+"_dataset/datasetval_"+parser.exp_type+"2_750_"+parser.bug_type+ "_" + str(parser.set_size) +".npy")
# y_val2 = np.load("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"2_750_"+parser.bug_type+"_" + str(parser.set_size) +".npy")

# X_val = np.concatenate((X_val, X_val1, X_val2))
# y_val = np.concatenate((y_val, y_val1, y_val2))

train_datagen = DataGenerator(img_files=X, labels=y, exp_type=parser.exp_type)
val_datagen = DataGenerator(img_files=X_val, labels=y_val, exp_type=parser.exp_type)


#new model

if parser.exp_type == 'baseline':
    vector_input = Input((10,))
    mlp_output = Dense(100, activation="relu")(vector_input)
    x2 = Concatenate()([final_output, mlp_output])
    x2 = Dense(100, activation="relu")(x2)
    predictions = Dense(1, activation= 'sigmoid')(x2)

    new_model = Model(inputs = [model.input, vector_input], outputs = predictions)
else:
    in2 = Input(shape=(224,224,3))
    x1 = base_model(in2)
    x1 = Flatten()(x1)
    x1 = Dropout(0.5)(x1)
    final_output1 = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001))(x1)

    x = Concatenate()([final_output, final_output1])
    x = Dense(100, activation="relu")(x)
    predictions = Dense(1, activation= 'sigmoid')(x)
    new_model = Model(inputs = [model.input, in2], outputs = predictions)

#Train model
batch_size = 16
from keras.optimizers import SGD, Adam
adam = Adam(lr=0.001)
new_model.compile(optimizer= adam, loss='binary_crossentropy', metrics=['accuracy'])

hist = new_model.fit_generator(train_datagen, 
                           steps_per_epoch=len(X) / batch_size, 
                           epochs=10, 
                           validation_data=val_datagen, 
                           validation_steps=1)