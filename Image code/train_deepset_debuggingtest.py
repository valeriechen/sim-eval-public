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
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D, SpatialDropout2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import regularizers
import tensorflow as tf
import keras

from gen_saliencymaps import *

os.system('rm -r temp_dog_dataset/')

n = 750

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

def load_image(path, size):
    ret = PIL.Image.open(path)
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    return ret

def extract_exp(exp):
  for key in exp.local_exp:
    temp = [item[1] for item in exp.local_exp[key]]
    return np.array(temp)

#FIGURE THIS
parser = argparse.ArgumentParser(description='DeepSets')
parser.add_argument('-d', '--dataset', dest='dataset', type=str, default=False,
                  help='Where to save generated data')
parser.add_argument('-b', '--bug', dest='bug_type', type=str, default=False,
                  help='What kind of bug')
parser.add_argument('-e', '--exp', dest='exp_type', type=str, default=False,
                  help='What kind of explanation')
parser.add_argument('-s', '--setsize', dest='set_size', type=int, default=False,
                  help='Size of set')
parser = parser.parse_args()


if 'baseline' in parser.exp_type:
  input_space = np.zeros((n, parser.set_size, 224, 224, 4))
  input_space_new = np.zeros((n, parser.set_size, 224, 224, 4))
else:
  input_space = np.zeros((n, parser.set_size, 224, 224, 6))
  input_space_new = np.zeros((n, parser.set_size, 224, 224, 6))
has_bugs = np.zeros((n,))

increments = 15

#make a list of all images across all 10 clasess.

for i in range(0,n,increments):

  #print(i)

  has_bug = True
  has_bugs[i] = 1

  if random.random() < 0.5:
    has_bug = False
    has_bugs[i] = 0

  if parser.bug_type == 'sc':
    train_it = load_debugging_tests_SC(has_bug)
  else:
    train_it = load_debugging_tests(has_bug, parser.bug_type)

  img_height,img_width = 224,224 
  num_classes = 10
  base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape= (img_height,img_width,3), )

  for layer in base_model.layers[:]:
     layer.trainable = False

  x = base_model.output
  x = Flatten()(x)
  x = Dropout(0.5)(x)
  x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001), name="dense_b")(x)
  x = Dropout(0.5)(x)
  predictions = Dense(num_classes, activation= 'softmax', name="dense_a")(x)
  model = Model(inputs = base_model.input, outputs = predictions)


  from keras.optimizers import SGD, Adam
  # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
  adam = Adam(lr=0.001)
  model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
  
  if parser.bug_type == 'sc':
    num_epochs = 1
    total_imgs = 1750
  else:
    num_epochs = 5
    total_imgs = 3346

  model.fit_generator(train_it, epochs = num_epochs, steps_per_epoch = total_imgs // 16)

  test_files = os.listdir(os.path.join("temp_dog_dataset", "test"))

  inds = np.random.choice(len(test_files), increments)  
  images = [load_image(os.path.join("temp_dog_dataset", "test", test_files[f]), 224) for f in inds]

  if 'baseline' in parser.exp_type:
    #get predictions
    all_input = np.zeros((len(images), 224, 224, 3))
    for t in range(len(images)):

        if images[t].shape[2] == 4:
          images[t] = cv2.cvtColor(images[t], cv2.COLOR_BGRA2BGR)

        all_input[t] = images[t]  
        all_input[t] = preprocess_input(all_input[t])  
    preds = model.predict(all_input)
  else:

    for t in range(len(images)):

        if images[t].shape[2] == 4:
          images[t] = cv2.cvtColor(images[t], cv2.COLOR_BGRA2BGR)
    
    exps = get_saliency_map('gradient', images, model, 'dog')

  for k in range(increments):
    print(i+k)
    
    if 'baseline' in parser.exp_type:
      actual_pred = np.argmax(preds[k]) / 10.
      exp = np.ones((224,224,1)) * actual_pred
      input_space[i+k][0] = np.concatenate([images[k], exp], axis=2) 
    else:
      input_space[i+k][0] = np.concatenate([images[k], exps[k]], axis=2) 
  
    if has_bug: #trying this instead of if sum is greater than 0
      has_bugs[i+k] = 1

  # if 'labelerror' in parser.bug_type: 
  #   test_files = os.listdir("dog_dataset_test")

  # inds = np.random.choice(len(test_files), increments)  
  # images = [load_image(os.path.join("dog_dataset_test", test_files[f]), 224) for f in inds]

  # if 'baseline' in parser.exp_type:
  #   #get predictions
  #   all_input = np.zeros((len(images), 224, 224, 3))
  #   for t in range(len(images)):
  #       all_input[t] = images[t]
  #       all_input[t] = preprocess_input(all_input[t])    
  #   preds = model.predict(all_input)
    
  # else:
    
  #   exps = get_saliency_map('gradient', images, model, 'dog')

  # for k in range(increments):
  #   print(i+k)
    
  #   if 'baseline' in parser.exp_type:
  #     actual_pred = np.argmax(preds[k]) / 10.
  #     exp = np.ones((224,224,1)) * actual_pred
  #     input_space_new[i+k][0] = np.concatenate([images[k], exp], axis=2) 
  #   else:
  #     input_space_new[i+k][0] = np.concatenate([images[k], exps[k]], axis=2) 


np.save("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", input_space)
#np.save("../all_datasets/"+parser.dataset+"_dataset/datasetval_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", input_space_new)
np.save("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", has_bugs)

