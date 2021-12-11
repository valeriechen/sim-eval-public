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


#this is for simulatability

def extract_exp(exp):
  for key in exp.local_exp:
    temp = [item[1] for item in exp.local_exp[key]]
    return np.array(temp)

def prob(data):
    return np.array(list(zip(1-model.predict(data),model.predict(data))))

def wrapped_fn(x):
  p = model.predict(x).reshape(-1, 1)
  return np.hstack((1-p, p))

def process_dice(x):
  cols = ['Age', 'Workclass', 'Education', 'Education-Num', 'MaritalStatus',
              'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week',
              'Country']

  exp = []
  for col in cols:
    exp.append(x[col])

  return exp

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

X,X_t,y= load_income_nobug()

#split into train/test/validation

X_train, X_test, X_t_train, X_t_test, y_train, y_test = train_test_split(X, X_t,y,test_size = 0.3,random_state=0)
X_test, X_val, X_t_test, X_t_val, y_test, y_val = train_test_split(X_test, X_t_test, y_test, test_size = 0.33, random_state=0)


#Verify X_val and y_val are the same

if parser.exp_type == 'gams':
  seed = 1
  ebm = ExplainableBoostingClassifier(random_state=seed)
  ebm.fit(X_train, y_train)
else:
  model = lgb.LGBMClassifier()
  model.fit(X_train, y_train)

if parser.exp_type == 'baseline':
  input_space = np.zeros((n,parser.set_size,num_feat+1))
else:
  input_space = np.zeros((n,parser.set_size,num_feat+num_feat))
has_bugs = np.zeros((n,))


if parser.exp_type == 'shap':
  explainer = shap.TreeExplainer(model)
elif parser.exp_type == 'lime':
  explainer = lime.lime_tabular.LimeTabularExplainer(X_train, discretize_continuous=True)
elif parser.exp_type == 'dice':
  explainer = dice_ml.Dice(d, m, method="random")
else:
  print("EXPLAINER NOT SPECIFIED")

for i in range(0,n):

  print(i)
  
  inds = np.random.choice(10000, parser.set_size)  

  for j in range(parser.set_size):        
    
    input_space[i][j][:num_feat] = X_t_train[inds[j]] #coordinates

    if parser.exp_type == 'baseline':
      temp = model.predict(X_train[inds[j]].reshape((1,num_feat)))
      input_space[i][j][-num_feat:] = temp
    elif parser.exp_type == 'shap':
      shap_values = explainer.shap_values(X_train[inds[j]].reshape((1,num_feat)), approximate=False, check_additivity=False)
      input_space[i][j][-num_feat:] = shap_values[1][0] #explanations
    elif parser.exp_type == 'lime':
      exp = explainer.explain_instance(X_train[inds[j]], wrapped_fn, num_features=13, top_labels=1, num_samples=50)
      input_space[i][j][-num_feat:] = extract_exp(exp)
    elif parser.exp_type == 'dice':
    
      curr_point = x_train[inds[j]:inds[j]+1]
      imp = explainer.local_feature_importance(curr_point, total_CFs=50) 

      while len(imp.local_importance) == 0:
        ind = np.random.choice(10000,1)
        curr_point = x_train[ind[0]:ind[0]+1]
        imp = explainer.local_feature_importance(curr_point, total_CFs=50) 

      input_space[i][j][-num_feat:] = process_dice(imp.local_importance[0])
      
    elif parser.exp_type == 'gams':
      ebm_local = ebm.explain_local(X_train[inds[j]:inds[j]+1], y_train[inds[j]:inds[j]+1])
      input_space[i][j][-num_feat:] = ebm_local._internal_obj['specific'][0]['scores'][:num_feat]

  if parser.exp_type == 'gams':
    has_bugs[i] = ebm.predict(X_train[inds[0]].reshape((1,num_feat)))
    print(has_bugs[i])
  else:
    has_bugs[i] = model.predict(X_train[inds[0]].reshape((1,num_feat)))#y[inds[0]] # instead of y use the predicted value!!
  
    # if sum(which_has_bugs[inds]) > 0:
    #   has_bugs[i+k] = 1

np.save("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", input_space)
np.save("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", has_bugs)


## GENERATE THE VALIDATION PART

n = 250

if parser.exp_type == 'baseline':
  input_space = np.zeros((n,parser.set_size,num_feat+1))
else:
  input_space = np.zeros((n,parser.set_size,num_feat+num_feat))
has_bugs = np.zeros((n,))

for i in range(0,n):

  print(i)
  
  inds = np.random.choice(3224, parser.set_size)  

  for j in range(parser.set_size):        
    
    input_space[i][j][:num_feat] = X_t_val[inds[j]] #coordinates

    if parser.exp_type == 'baseline':
      temp = model.predict(X_val[inds[j]].reshape((1,num_feat)))
      input_space[i][j][-num_feat:] = temp
    elif parser.exp_type == 'shap':
      shap_values = explainer.shap_values(X_val[inds[j]].reshape((1,num_feat)), approximate=False, check_additivity=False)
      input_space[i][j][-num_feat:] = shap_values[1][0] #explanations
    elif parser.exp_type == 'lime':
      exp = explainer.explain_instance(X_val[inds[j]], wrapped_fn, num_features=13, top_labels=1, num_samples=50)
      input_space[i][j][-num_feat:] = extract_exp(exp)
    elif parser.exp_type == 'gams':
      ebm_local = ebm.explain_local(X_val[inds[j]:inds[j]+1], y_val[inds[j]:inds[j]+1])
      input_space[i][j][-num_feat:] = ebm_local._internal_obj['specific'][0]['scores'][:num_feat]

  if parser.exp_type == 'gams':
    has_bugs[i] = ebm.predict(X_val[inds[0]].reshape((1,num_feat)))
    print(has_bugs[i])
  elsed:
    has_bugs[i] = model.predict(X_val[inds[0]].reshape((1,num_feat)))#y[inds[0]] # instead of y use the predicted value!!
  

np.save("../all_datasets/"+parser.dataset+"_dataset/datasetval_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", input_space)
np.save("../all_datasets/"+parser.dataset+"_dataset/bugsval_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", has_bugs)


