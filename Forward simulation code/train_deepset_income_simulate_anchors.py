import random
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from numpy import random as random1
import shap
import lime
import matplotlib.pyplot as plt 
import argparse
from load_datasets import *
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import dice_ml
from dice_ml.utils import helpers
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from anchor import utils
from anchor import anchor_tabular

#this is for simulatability

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
parser.add_argument('-n', '--tot', dest='tot_runs', type=int, default=False,
                  help='Size of set')
parser = parser.parse_args()

n = parser.tot_runs
num_feat = 13

from __future__ import print_function
import numpy as np
np.random.seed(1)
import sys
import sklearn
import sklearn.ensemble
from anchor import utils
from anchor import anchor_tabular
import pandas as pd
import copy
import sklearn
import numpy as np
import lime
import lime.lime_tabular
# import string
import os
import sys

class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)

def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_features=None,
                     features_to_use=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_features, consider everything < 20 as categorical"""
    if feature_transformations is None:
        feature_transformations = {}
    try:
        data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
    except:
        import pandas
        data = pandas.read_csv(data,
                               header=None,
                               delimiter=delimiter,
                               na_filter=True,
                               dtype=str).fillna(fill_na).values


    if target_idx < 0:
        target_idx = data.shape[1] + target_idx
    ret = Bunch({})
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])

    labels = data[:, target_idx]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    labels = ret.labels
    ret.class_names = list(le.classes_)
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_features:
            categorical_features = ([x if x < target_idx else x - 1
                                     for x in categorical_features])
    if categorical_features is None:
        categorical_features = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_features.append(f)
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)
    ordinal_features = []
    if discretize:
        disc = lime.lime_tabular.QuartileDiscretizer(data,
                                                     categorical_features,
                                                     feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]
        categorical_features = list(range(data.shape[1]))
        categorical_names.update(disc.names)
    for x in categorical_names:
        categorical_names[x] = [y.decode() if type(y) == np.bytes_ else y for y in categorical_names[x]]
    ret.ordinal_features = ordinal_features
    ret.categorical_features = categorical_features
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                  test_size=.2,
                                                  random_state=1)
    train_idx, test_idx = [x for x in splits.split(data)][0]
    ret.train = data[train_idx]
    ret.labels_train = ret.labels[train_idx]
    cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                     test_size=.5,
                                                     random_state=1)
    cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
    cv_idx = test_idx[cv_idx]
    test_idx = test_idx[ntest_idx]

    ret.validation = data[cv_idx]
    ret.labels_validation = ret.labels[cv_idx]
    ret.test = data[test_idx]
    ret.labels_test = ret.labels[test_idx]
    ret.test_idx = test_idx
    ret.validation_idx = cv_idx
    ret.train_idx = train_idx

    # ret.train, ret.test, ret.labels_train, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(data, ret.labels,
    #                                               train_size=0.80))
    # ret.validation, ret.test, ret.labels_validation, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(ret.test, ret.labels_test,
    #                                               train_size=.5))
    ret.data = data

    return ret


feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                         "Education-Num", "MaritalStatus", "Occupation",
                         "Relationship", "Race", "Sex", "CapitalGain",
                         "CapitalLoss", "Hoursperweek", "Country", 'Income']
features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
categorical_features = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
education_map = {
    '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
    'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
    'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
    'Some-college': 'High School grad', 'Masters': 'Masters',
    'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
    'Assoc-voc': 'Associates',
}
occupation_map = {
    "Adm-clerical": "Admin", "Armed-Forces": "Military",
    "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
    "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
    "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
    "Service", "Priv-house-serv": "Service", "Prof-specialty":
    "Professional", "Protective-serv": "Other", "Sales":
    "Sales", "Tech-support": "Other", "Transport-moving":
    "Blue-Collar",
}
country_map = {
    'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
    'China', 'Columbia': 'South-America', 'Cuba': 'Other',
    'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
    'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
    'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
    'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
    'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
    'Hong': 'China', 'Hungary': 'Euro_2', 'India':
    'British-Commonwealth', 'Iran': 'Other', 'Ireland':
    'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
    'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
    'Latin-America', 'Nicaragua': 'Latin-America',
    'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
    'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
    'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
    'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
    'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
    'United-States': 'United-States', 'Vietnam': 'SE-Asia'
}
married_map = {
    'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
    'Married-civ-spouse': 'Married', 'Married-spouse-absent':
    'Separated', 'Separated': 'Separated', 'Divorced':
    'Separated', 'Widowed': 'Widowed'
}
label_map = {'<=50K': 'Less than $50,000', '>50K': 'More than $50,000'}

def cap_gains_fn(x):
    x = x.astype(float)
    d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                    right=True).astype('|S128')
    return map_array_values(d, {'0': 'None', '1': 'Low', '2': 'High'})

transformations = {}

# transformations = {
#     3: lambda x: map_array_values(x, education_map),
#     5: lambda x: map_array_values(x, married_map),
#     6: lambda x: map_array_values(x, occupation_map),
#     10: cap_gains_fn,
#     11: cap_gains_fn,
#     13: lambda x: map_array_values(x, country_map),
#     14: lambda x: map_array_values(x, label_map),
# }
dataset = load_csv_dataset(
    'adult.data', -1, ', ',
    feature_names=feature_names, features_to_use=features_to_use,
    categorical_features=categorical_features, discretize=False,
    balance=False, feature_transformations=transformations)

#dataset = utils.load_dataset('adult', balance=True, dataset_folder='datasets', discretize=True)

#print(dataset.train)

# print(dataset.train[:,0])
#c = lgb.LGBMClassifier()
#c.fit(dataset.train, dataset.labels_train)
c = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
c.fit(dataset.train, dataset.labels_train)
print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, c.predict(dataset.train)))
print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, c.predict(dataset.test)))

explainer = anchor_tabular.AnchorTabularExplainer(
    dataset.class_names,
    dataset.feature_names,
    dataset.train,
    dataset.categorical_names)


dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("MaritalStatus", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("CapitalGain", "float32"), ("CapitalLoss", "float32"),
        ("Hoursperweek", "float32"), ("Country", "category"), ("Target", "category")
    ]
data = pd.read_csv(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
  names=[d[0] for d in dtypes],
      dtype=dict(dtypes))
filt_dtypes = list(filter(lambda x: not (x[0] in ["Target"]), dtypes))
  y = data["Target"] == " >50K"
  y = LabelEncoder().fit_transform(y)

rcode = {
      "Not-in-family": 0,
      "Unmarried": 1,
      "Other-relative": 2,
      "Own-child": 3,
      "Husband": 4,
      "Wife": 5
}
categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_

data = data.drop(["Target", "fnlwgt"], axis=1)


#addd


#GENERATE EXPLANATIONS FOR TRAIN AND VALIDATION
ids = np.random.choice(X_train.shape[0], n)

has_bugs = np.zeros((n,))

for i in ids:
  exp = explainer.explain_instance(dataset.train[i], c.predict, threshold=0.95)
  
  new_item = [-1.0, None, None, -1.0, None, None, None, None, None, -1.0, -1.0, -1.0, None]

  for item in exp.names():
    #split using white space
    parts = item.split()
    if dtypes[feature_names.index(parts[0])][1] == "float32":
      new_item[feature_names.index(parts[0])] = float(parts[2])
    else:
      new_item[feature_names.index(parts[0])] = parts[2]

  #save model prediction
  has_bugs =  model.predict(X_train[i].reshape((1,num_feat)))

  df_length = len(data)
  data.loc[df_length] = new_item

ids = np.random.choice(X_val.shape[0], 250)

input_space_val = np.zeros((n,num_feat+num_feat))
has_bugs_val = np.zeros((n,))

for i in ids:
  exp = explainer.explain_instance(dataset.validation[i], c.predict, threshold=0.95)
  
  new_item = [-1.0, None, None, -1.0, None, None, None, None, None, -1.0, -1.0, -1.0, None]

  for item in exp.names():
    #split using white space
    parts = item.split()
    if dtypes[feature_names.index(parts[0])][1] == "float32":
      new_item[feature_names.index(parts[0])] = float(parts[2])
    else:
      new_item[feature_names.index(parts[0])] = parts[2]

  #save model prediction
  has_bugs_val =  model.predict(X_train[i].reshape((1,num_feat)))

  df_length = len(data)
  data.loc[df_length] = new_item


#Transform into labels, then min-max it

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


X = data
X = X.values

scaler = MinMaxScaler()
train_X_scaled = scaler.fit_transform(X)

#fill in dataset and dataset_val




#-------------------

# X_train, X_test, X_t_train, X_t_test, y_train, y_test = train_test_split(X, X_t,y,test_size = 0.3,random_state=0)
# X_test, X_val, X_t_test, X_t_val, y_test, y_val = train_test_split(X_test, X_t_test, y_test, test_size = 0.33, random_state=0)




# input_space = np.zeros((n,parser.set_size,num_feat+num_feat))
# has_bugs = np.zeros((n,))


# if parser.exp_type == 'shap':
#   explainer = shap.TreeExplainer(model)
# elif parser.exp_type == 'lime':
#   explainer = lime.lime_tabular.LimeTabularExplainer(X_train, discretize_continuous=True)
# elif parser.exp_type == 'dice':
#   explainer = dice_ml.Dice(d, m, method="random")
# else:
#   print("EXPLAINER NOT SPECIFIED")

# for i in range(0,n):

#   print(i)
  
#   inds = np.random.choice(10000, parser.set_size)  

#   for j in range(parser.set_size):        
    
#     input_space[i][j][:num_feat] = X_t_train[inds[j]] #coordinates

#     if parser.exp_type == 'baseline':
#       temp = model.predict(X_train[inds[j]].reshape((1,num_feat)))
#       input_space[i][j][-num_feat:] = temp
#     elif parser.exp_type == 'shap':
#       shap_values = explainer.shap_values(X_train[inds[j]].reshape((1,num_feat)), approximate=False, check_additivity=False)
#       input_space[i][j][-num_feat:] = shap_values[1][0] #explanations
#     elif parser.exp_type == 'lime':
#       exp = explainer.explain_instance(X_train[inds[j]], wrapped_fn, num_features=13, top_labels=1, num_samples=50)
#       input_space[i][j][-num_feat:] = extract_exp(exp)
#     elif parser.exp_type == 'dice':
    
#       curr_point = x_train[inds[j]:inds[j]+1]
#       imp = explainer.local_feature_importance(curr_point, total_CFs=50) 

#       while len(imp.local_importance) == 0:
#         ind = np.random.choice(10000,1)
#         curr_point = x_train[ind[0]:ind[0]+1]
#         imp = explainer.local_feature_importance(curr_point, total_CFs=50) 

#       input_space[i][j][-num_feat:] = process_dice(imp.local_importance[0])
      
#     elif parser.exp_type == 'gams':
#       ebm_local = ebm.explain_local(X_train[inds[j]:inds[j]+1], y_train[inds[j]:inds[j]+1])
#       input_space[i][j][-num_feat:] = ebm_local._internal_obj['specific'][0]['scores'][:num_feat]

#   has_bugs[i] = model.predict(X_train[inds[0]].reshape((1,num_feat)))#y[inds[0]] # instead of y use the predicted value!!

#     # if sum(which_has_bugs[inds]) > 0:
#     #   has_bugs[i+k] = 1

# np.save("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", input_space)
# np.save("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", has_bugs)


# ## GENERATE THE VALIDATION PART

# n = 250

# if parser.exp_type == 'baseline':
#   input_space = np.zeros((n,parser.set_size,num_feat+1))
# else:
#   input_space = np.zeros((n,parser.set_size,num_feat+num_feat))
# has_bugs = np.zeros((n,))

# for i in range(0,n):

#   print(i)
  
#   inds = np.random.choice(3224, parser.set_size)  

#   for j in range(parser.set_size):        
    
#     input_space[i][j][:num_feat] = X_t_val[inds[j]] #coordinates

#     if parser.exp_type == 'baseline':
#       temp = model.predict(X_val[inds[j]].reshape((1,num_feat)))
#       input_space[i][j][-num_feat:] = temp # check this, this leaks the answer!!
#     elif parser.exp_type == 'shap':
#       shap_values = explainer.shap_values(X_val[inds[j]].reshape((1,num_feat)), approximate=False, check_additivity=False)
#       input_space[i][j][-num_feat:] = shap_values[1][0] #explanations
#     elif parser.exp_type == 'lime':
#       exp = explainer.explain_instance(X_val[inds[j]], wrapped_fn, num_features=13, top_labels=1, num_samples=50)
#       input_space[i][j][-num_feat:] = extract_exp(exp)
#     elif parser.exp_type == 'gams':
#       ebm_local = ebm.explain_local(X_val[inds[j]:inds[j]+1], y_val[inds[j]:inds[j]+1])
#       input_space[i][j][-num_feat:] = ebm_local._internal_obj['specific'][0]['scores'][:num_feat]

#   has_bugs[i] = model.predict(X_val[inds[0]].reshape((1,num_feat)))#y[inds[0]] 


# np.save("../all_datasets/"+parser.dataset+"_dataset/datasetval_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", input_space)
# np.save("../all_datasets/"+parser.dataset+"_dataset/bugsval_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", has_bugs)


