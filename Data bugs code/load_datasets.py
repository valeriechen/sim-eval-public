import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_debugging_tests_labelerror(has_bug, bug_type):

  from keras.preprocessing.image import ImageDataGenerator 
  from keras.applications.resnet50 import preprocess_input, decode_predictions
  import os
  from random import sample

  #make a copy of the unsplit dataset
  os.system('rm -r temp_dog_dataset/')
  os.system('mkdir temp_dog_dataset')
  os.system('cp -r dog_dataset_train temp_dog_dataset/train')
  os.system('mkdir temp_dog_dataset/test')

  #randomly split into train and test

  breeds = ['Beagle', 'Boxer', 'Chihuahua', 'GreatPyrenees', 'Newfoundland', 'Pomeranian', 'Pug', 'SaintBernard', 'WheatenTerrier', 'YorkshireTerrier']

  #for breed in breeds:
  #  os.system('mkdir temp_dog_dataset/test/'+breed)

  if has_bug:

    if bug_type == 'OOD':

      for breed in breeds:
        
        files = os.listdir('temp_dog_dataset/train/'+breed)
        for file in sample(files,25):
          os.system('mv temp_dog_dataset/train/'+breed + '/'+file + ' temp_dog_dataset/test') 

      #add some random breeds to tempdogdataset/test

      #only this one for test?
      os.system('cp -a ../OOD_dogs/. temp_dog_dataset/test')

    elif 'labelerror' in bug_type:

      #ranodmly shuffle images between folders

      num_move = int(bug_type[-3:]) #200 (200/400 -- 50%)
      print("num move is ", num_move)

      for breed in breeds:
        files = os.listdir('temp_dog_dataset/train/'+breed)
        for file in sample(files,num_move):
          ind = random.randint(0,9)
          if breeds[ind] != breed:
            os.system('mv temp_dog_dataset/train/'+breed + '/' + file + ' temp_dog_dataset/train/'+breeds[ind])
          #if 'Beagle' != breed:
          #  os.system('mv temp_dog_dataset/train/'+breed + '/' + file + ' temp_dog_dataset/train/Beagle')

  if 'labelerror' in bug_type:
   
    for breed in breeds:
        files = os.listdir('temp_dog_dataset/train/'+breed)
        for file in sample(files,25):
          os.system('mv temp_dog_dataset/train/'+breed + '/' + file + ' temp_dog_dataset/test/')           
          #os.system('mv temp_dog_dataset/train/'+breed + '/' + file + ' temp_dog_dataset/test/'+breed + '/') 

  #subset part of the dataset??
  datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

  train_it = datagen.flow_from_directory('temp_dog_dataset/train/', target_size =(224,224), class_mode='categorical') # or should this be categorical
  #test_it = datagen.flow_from_directory('temp_dog_dataset/test/', target_size =(224,224), class_mode='categorical')

  return train_it#, test_it


def load_debugging_tests_SC(has_bug):

  from keras.preprocessing.image import ImageDataGenerator 
  from keras.applications.resnet50 import preprocess_input, decode_predictions
  import os
  from random import sample

  #make a copy of the unsplit dataset
  os.system('rm -r temp_dog_dataset/')
  os.system('mkdir temp_dog_dataset')
  os.system('mkdir temp_dog_dataset/test')

  #randomly split into train and test

  breeds = ['Beagle', 'Boxer', 'Chihuahua', 'GreatPyrenees', 'Newfoundland', 'Pomeranian', 'Pug', 'SaintBernard', 'WheatenTerrier', 'YorkshireTerrier']

  if has_bug:
    os.system('cp -r new_Dog_dataset temp_dog_dataset/train')
  else:
    os.system('cp -r new_Dog_dataset_SC temp_dog_dataset/train')
    
  for breed in breeds:
      files = os.listdir('temp_dog_dataset/train/'+breed)
      for file in sample(files,25):
        os.system('mv temp_dog_dataset/train/'+breed + '/' + file + ' temp_dog_dataset/test/')           

  #subset part of the dataset??
  datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

  train_it = datagen.flow_from_directory('temp_dog_dataset/train/', target_size =(224,224), class_mode='categorical') # or should this be categorical
  #test_it = datagen.flow_from_directory('temp_dog_dataset/test/', target_size =(224,224), class_mode='categorical')

  return train_it#, test_it


def load_income_CHI_new(has_bug,bug_type):
  
  # dtypes = [
  #       ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
  #       ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
  #       ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
  #       ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
  #       ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
  #   ]
  # data = pd.read_csv(
  #     "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
  #     names=[d[0] for d in dtypes],
  #         dtype=dict(dtypes))

  # filt_dtypes = list(filter(lambda x: not (x[0] in ["Target"]), dtypes))
  # y = data["Target"] == " >50K"
  # y = LabelEncoder().fit_transform(y)
  # data = data.drop(["Target", "fnlwgt"], axis=1)

  # dataframe = pd.read_csv( "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None, na_values='?')
  # dataframe = dataframe.dropna()
  # last_ix = len(dataframe.columns) - 1
  # X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
  # X.columns = ['Age', 'Workclass', 'FinalWeight', 'Education', 'EducationNumberOfYears', 'Marital-status',
  #             'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week',
  #             'Native-country']
  # X = X.drop(["FinalWeight"], axis=1)

  # cat_ix = X.select_dtypes(include=['object', 'bool']).columns
  # num_ix = X.select_dtypes(include=['int64', 'float64']).columns

  # encoder = OneHotEncoder()
  # train_X_encoded = encoder.fit_transform(X[cat_ix])
  # train_X_encoded = train_X_encoded.toarray()
  # column_name = encoder.get_feature_names(cat_ix)

  # scaler = MinMaxScaler()
  # train_X_scaled = scaler.fit_transform(X[num_ix])

  # X_comb = np.concatenate((train_X_encoded, train_X_scaled), axis=1)

  # print(X_comb.shape)
  # X = X_comb

  dataframe = pd.read_csv('../base_data/adult-all.csv', header=None, na_values='?')
  dataframe = dataframe.dropna()
  last_ix = len(dataframe.columns) - 1
  X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
  X.columns = ['Age', 'Workclass', 'FinalWeight', 'Education', 'EducationNumberOfYears', 'Marital-status',
              'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week',
              'Native-country']
  X = X.drop(["FinalWeight"], axis=1)

  cat_ix = X.select_dtypes(include=['object', 'bool']).columns
  num_ix = X.select_dtypes(include=['int64', 'float64']).columns
  y = LabelEncoder().fit_transform(y)

  encoder = OneHotEncoder()
  train_X_encoded = encoder.fit_transform(X[cat_ix])
  train_X_encoded = train_X_encoded.toarray()
  column_name = encoder.get_feature_names(cat_ix)

  scaler = MinMaxScaler()
  train_X_scaled = scaler.fit_transform(X[num_ix])

  X_comb = np.concatenate((train_X_scaled, train_X_encoded), axis=1)

  # print(X_comb.shape)

  if bug_type == 'missingvalue':

    #random.shuffle(X_comb[:,2])
    #ind_greater_50k = [i for i in range(len(y==1))] 
    ind_greater_50k = []
    for t in range(len(y)):
      if y[t]==1:
        ind_greater_50k.append(t)
    inds = np.random.choice(ind_greater_50k, int(0.1*len(ind_greater_50k))).tolist()
    X_comb[inds,0] = 38.0

  elif bug_type == 'redundantfeatures':
    #if not bugged, replace education with a randomly shuffled vector so they are not correlated.
    if not has_bug:
      random.shuffle(X_comb[:,2]) #this is the education column
    else:
      #90% are randomized, only 10% have redundant features
      ind_greater_50k = [i for i in range(len(y))] 
      inds = np.random.choice(ind_greater_50k, int(0.5*len(ind_greater_50k))).tolist()
      temp = np.copy(X_comb[:,2])
      random.shuffle(X_comb[:,2])
      X_comb[inds,2] = temp[inds]

  elif bug_type == 'randomlabels' and has_bug:

    print(y)

    inds = np.random.choice(X_comb.shape[0], int(0.1*X_comb.shape[0])).tolist()
    for i in range(len(inds)):
      y[inds[i]] = random.choice([0., 1.])
    #random.shuffle(y)

  inds = np.random.choice(X_comb.shape[0], 10000)  
  X_subset = X_comb[inds,:]
  y_subset = y[inds]

  inds = np.random.choice(X_comb.shape[0], 1500)  
  X_subset1 = X_comb[inds,:]
  y_subset1 = y[inds]

  return X_subset, y_subset, X_subset1, y_subset1


def load_income_CHI(has_bug,bug_type):
  
  dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
  data = pd.read_csv(
      "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
      names=[d[0] for d in dtypes],
          dtype=dict(dtypes))

  filt_dtypes = list(filter(lambda x: not (x[0] in ["Target"]), dtypes))
  y = data["Target"] == " >50K"
  y = LabelEncoder().fit_transform(y)

  data= data.drop(["Target", "fnlwgt"], axis=1)
  rcode = {
      "Not-in-family": 0,
      "Unmarried": 1,
      "Other-relative": 2,
      "Own-child": 3,
      "Husband": 4,
      "Wife": 5
  }
  for k, dtype in filt_dtypes:
      if dtype == "category":
          if k == "Relationship":
              data[k] = np.array([rcode[v.strip()] for v in data[k]])
          else:
              data[k] = data[k].cat.codes

  #data = data.drop(["Target", "fnlwgt", "Hours per week", "Capital Loss", "Workclass", "Education-Num", "Relationship", "Race", "Sex", "Capital Gain"], axis=1)

  #print(data.columns)

  X = data
  X = X.values

  if has_bug and 'missingvalue' in bug_type:

    # random.shuffle(X[:,2])
    ind_greater_50k = []
    for t in range(len(y)):
      if y[t]==1:
        ind_greater_50k.append(t)

    perc = float(bug_type[-3:])
    print(perc)
    inds = np.random.choice(ind_greater_50k, int(perc*len(ind_greater_50k))).tolist()
    #print(np.mean(X[:,0]))
    X[inds,0] = 38.0
    #print(np.mean(X[:,0]))

  elif bug_type == 'redundantfeatures':
    #if not bugged, replace education with a randomly shuffled vector so they are not correlated.
    if not has_bug:
      random.shuffle(X[:,2]) #this is the education column
    else:
      #90% are randomized, only 10% have redundant features
      ind_greater_50k = [i for i in range(len(y))] 
      inds = np.random.choice(ind_greater_50k, int(0.1*len(ind_greater_50k))).tolist()
      temp = np.copy(X[:,2])
      random.shuffle(X[:,2])
      X[inds,2] = temp[inds]

  elif 'randomlabels' in bug_type and has_bug:
    
    ind_greater_50k = []
    for t in range(len(y)):
      if y[t]==1:
        ind_greater_50k.append(t)
    inds = np.random.choice(ind_greater_50k, int(0.1*len(ind_greater_50k))).tolist()
    y[inds] = 0

    #ADDS BUGS TO EVERYTHING
    #inds = np.random.choice(X.shape[0], int(0.1*X.shape[0])).tolist()
    #for i in range(len(inds)):
    #  y[inds[i]] = random.choice([0, 1])


  scaler = MinMaxScaler()
  train_X_scaled = scaler.fit_transform(X)

  inds = np.random.choice(X.shape[0], 10000)  
  X_subset = X[inds,:]
  y_subset = y[inds]
  X_t_subset = train_X_scaled[inds,:]

  inds = np.random.choice(X.shape[0], 1500)  
  X_subset1 = X[inds,:]
  y_subset1 = y[inds]


  return X_subset, y_subset, X_subset1, y_subset1, X_t_subset

def load_income_nobug():
  dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
  data = pd.read_csv(
      "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
      names=[d[0] for d in dtypes],
          dtype=dict(dtypes))

  filt_dtypes = list(filter(lambda x: not (x[0] in ["Target"]), dtypes))
  y = data["Target"] == " >50K"
  y = LabelEncoder().fit_transform(y)

  data= data.drop(["Target", "fnlwgt"], axis=1)
  rcode = {
      "Not-in-family": 0,
      "Unmarried": 1,
      "Other-relative": 2,
      "Own-child": 3,
      "Husband": 4,
      "Wife": 5
  }
  for k, dtype in filt_dtypes:
      if dtype == "category":
          if k == "Relationship":
              data[k] = np.array([rcode[v.strip()] for v in data[k]])
          else:
              data[k] = data[k].cat.codes

  #data = data.drop(["Target", "fnlwgt", "Hours per week", "Capital Loss", "Workclass", "Education-Num", "Relationship", "Race", "Sex", "Capital Gain"], axis=1)

  #print(data.columns)

  X = data
  X = X.values

  scaler = MinMaxScaler()
  train_X_scaled = scaler.fit_transform(X)

  return X, train_X_scaled, y


#---------

def inject_bug(X,y, bug_type, which_has_bugs, means, variances):
  #ind_greater_50k = np.where(y==1)[0].tolist()
  ind_greater_50k = [i for i in range(len(y))] 
  inds = np.random.choice(ind_greater_50k, int(0.25*len(ind_greater_50k))).tolist()
  which_has_bugs[inds] = 1
  if bug_type == 'labelleak':
    X[inds,0] = y[inds]
  elif bug_type == 'corruptfeat':
    X[inds,0] = -1.
  elif bug_type == 'labelerror':
    y[inds] = 0 # would need to change this for multi-class... 
  elif bug_type == 'dupfeat':
    X[inds,0] = X[inds,1]
  elif bug_type == 'ood':

    new_data = np.zeros((len(inds),3))
    for i in range(3):
      max_val = means[i]
      new_data[:,i] =max_val*np.ones(len(inds),) + np.random.uniform(-0.5*variances[i],0.5*variances[i], (len(inds),))
    
    X[inds] = new_data
    y[inds] = 1

  return X, which_has_bugs

def get_dataset_stats(X):

  means = []
  variances = []

  for i in range(X.shape[1]):
    means.append(max(X[:,i]))
    #means.append(np.mean(X[:,i]))
    variances.append(np.var(X[:,i]))

  return means, variances

def load_income_dataset(has_bug, bug_type=None):
	# load the dataset as a numpy array
	dataframe = pd.read_csv('../base_data/adult-all.csv', header=None, na_values='?')
  
	# drop rows with missing
	dataframe = dataframe.dropna()
	# split into inputs and outputs
	last_ix = len(dataframe.columns) - 1
	X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
	X.columns = ['Age', 'Workclass', 'FinalWeight', 'Education', 'EducationNumberOfYears', 'Marital-status',
              'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week',
              'Native-country']
	cat_ix = X.select_dtypes(include=['object', 'bool']).columns
	num_ix = X.select_dtypes(include=['int64', 'float64']).columns

	scaler = MinMaxScaler()
	train_X_scaled = scaler.fit_transform(X[num_ix]);  means, variances = get_dataset_stats(train_X_scaled) # for each feat

	train_X_new = SelectKBest(chi2, k=3).fit_transform(train_X_scaled, y)  
	y = LabelEncoder().fit_transform(y)
	which_has_bugs = np.zeros((train_X_new.shape[0],))
	if has_bug:
		train_X_new, which_has_bugs = inject_bug(train_X_new,y, bug_type, which_has_bugs, means, variances)

	inds = np.random.choice(train_X_new.shape[0], 10000)  
	X_subset = train_X_new[inds,:]
	y_subset = y[inds]
	which_has_bugs = which_has_bugs[inds]

	return X_subset, y_subset, which_has_bugs

def load_xor_dataset(has_bug,bug_type=None):
  n=2500
  dataset = np.random.rand(n,3)
  labels = np.zeros(n)

  for i in range(n):
    if dataset[i][0] > 0.5 and dataset[i][1] > 0.5:
      labels[i] = 1.
    elif dataset[i][0] < 0.5 and dataset[i][1] < 0.5:
      labels[i] = 1.
    else:
      labels[i] = 0.

  means, variances = get_dataset_stats(dataset) # for each feat

  which_has_bugs = np.zeros((dataset.shape[0],))
  if has_bug:
    dataset, which_has_bugs = inject_bug(dataset,labels, bug_type,which_has_bugs,means, variances)

  # scaler = MinMaxScaler()
  # dataset = scaler.fit_transform(dataset)

  inds = np.random.choice(dataset.shape[0], 1000)  
  X_subset = dataset[inds,:]
  y_subset = labels[inds]
  which_has_bugs = which_has_bugs[inds]

  return dataset, labels, which_has_bugs

def load_wifi_dataset(has_bug, bug_type=None):
  data = np.loadtxt('../base_data/wifi_localization.txt', delimiter='\t')
  X = data[:,:-1]
  y = data[:,-1]
  scaler = MinMaxScaler() ## SHOULD THIS COME BEFORE OR AFTER?
  train_X_scaled = scaler.fit_transform(X)
  y = LabelEncoder().fit_transform(y)

  means, variances = get_dataset_stats(train_X_scaled)

  train_X_new = SelectKBest(chi2, k=3).fit_transform(train_X_scaled, y) # just pick these 3

  which_has_bugs = np.zeros((train_X_new.shape[0],))
  if has_bug:
    train_X_new, which_has_bugs = inject_bug(train_X_new,y, bug_type,which_has_bugs,means, variances)

  inds = np.random.choice(X.shape[0], 1000)  
  X_subset = train_X_new[inds,:]
  y_subset = y[inds]
  which_has_bugs = which_has_bugs[inds]

  return X_subset, y_subset, which_has_bugs
