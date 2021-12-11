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
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from interpret.glassbox import ExplainableBoostingClassifier


#this is for bugs

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


if parser.exp_type == 'baseline':
  input_space = np.zeros((n,parser.set_size,num_feat+1))
else:
  input_space = np.zeros((n,parser.set_size,num_feat+num_feat))
has_bugs = np.zeros((n,))

for i in range(0,n,10):

  #print(i)

  has_bug = True
  has_bugs[i] = 1

  if random.random() < 0.5:
    has_bug = False
    has_bugs[i] = 0

  if parser.dataset == 'income_easy1':
    X_train,y_train, X_test, y_test = load_income_CHI_new(has_bug,parser.bug_type)

    # model = lgb.LGBMClassifier()
    # model.fit(X_train, y_train)

    # y_pred=model.predict(X_test)

    # accuracy=accuracy_score(y_pred, y_test)
    # print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))

    model = RandomForestClassifier(n_estimators=10).fit(X_train,y_train)

    # print("train accuracy", sklearn.metrics.accuracy_score(y_train, rf.predict(X_train)))
    # print("test accuracy", sklearn.metrics.accuracy_score(y_test, rf.predict(X_test)))

    # d_train = lgb.Dataset(X_train, label=y_train)
    # d_test = lgb.Dataset(X_test, label=y_test)

    # params = {
    #     "max_bin": 512,
    #     "learning_rate": 0.05,
    #     "boosting_type": "gbdt",
    #     "objective": "binary",
    #     "metric": "binary_logloss",
    #     "num_leaves": 10,
    #     "verbose": -1,
    #     "min_data": 100,
    #     "boost_from_average": True
    # }

    # model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
    # print(model.predict(X_test), y_test)
  elif parser.dataset=='income':
    X_train,y_train, X_test, y_test, X_t_train = load_income_CHI(has_bug,parser.bug_type)
    
    if parser.exp_type == 'gams':
      seed = 1
      ebm = ExplainableBoostingClassifier(random_state=seed)
      ebm.fit(X_train, y_train)
    elif parser.exp_type == 'dice':
      y_temp = np.reshape(y_train, (y_train.shape[0],1))
      dataset = np.concatenate((X_train, y_temp), axis=1)
      df = pd.DataFrame(data=dataset)

      df.columns = ['Age', 'Workclass', 'Education', 'Education-Num', 'MaritalStatus',
                    'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per week',
                    'Country', 'y']

      d = dice_ml.Data(dataframe=df, continuous_features=['Age', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week'], outcome_name='y')


      target = df["y"]
      datasetX = df.drop("y", axis=1)
      x_train, x_test, y_train, y_test = train_test_split(datasetX, 
                                                          target, 
                                                          test_size = 0.2,
                                                          random_state=0,
                                                          stratify=target)

      numerical = ['Age', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']
      categorical = x_train.columns.difference(numerical)

      categorical_transformer = Pipeline(steps=[
          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

      transformations = ColumnTransformer(
          transformers=[
              ('cat', categorical_transformer, categorical)])

      # Append classifier to preprocessing pipeline.
      # Now we have a full prediction pipeline.
      clf = Pipeline(steps=[('preprocessor', transformations),
                            ('classifier', lgb.LGBMClassifier())])
      model = clf.fit(x_train, y_train)

      m = dice_ml.Model(model=model, backend="sklearn")
    else:
      model = lgb.LGBMClassifier()
      model.fit(X_train, y_train)
  else:
    print("DATASET NOT DEFINED!")


  if parser.exp_type == 'shap':
    explainer = shap.TreeExplainer(model)
  elif parser.exp_type == 'lime':
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, discretize_continuous=True)
  elif parser.exp_type == 'dice':
    explainer = dice_ml.Dice(d, m, method="random")
  else:
    print("EXPLAINER NOT SPECIFIED")


  for k in range(10):
    print(i+k)
    inds = np.random.choice(10000, parser.set_size)  

    for j in range(parser.set_size):        
      
      input_space[i+k][j][:num_feat] = X_t_train[inds[j]] #coordinates


      if parser.exp_type == 'baseline':
        temp = model.predict(X_train[inds[j]].reshape((1,num_feat)))
        input_space[i+k][j][-num_feat:] = temp
      elif parser.exp_type == 'shap':
        shap_values = explainer.shap_values(X_train[inds[j]].reshape((1,num_feat)), approximate=False, check_additivity=False)
        input_space[i+k][j][-num_feat:] = shap_values[1][0] #explanations
      elif parser.exp_type == 'lime':
        curr_point = np.array([X_train[inds[j]]])#.reshape((1,num_feat))
        exp = explainer.explain_instance(X_train[inds[j]], wrapped_fn, num_features=13, top_labels=1, num_samples=50)
        input_space[i+k][j][-num_feat:] = extract_exp(exp)
      elif parser.exp_type == 'dice':
      
        curr_point = x_train[inds[j]:inds[j]+1]
        imp = explainer.local_feature_importance(curr_point, total_CFs=50) 

        while len(imp.local_importance) == 0:
          ind = np.random.choice(10000,1)
          curr_point = x_train[ind[0]:ind[0]+1]
          imp = explainer.local_feature_importance(curr_point, total_CFs=50) 

        input_space[i+k][j][-num_feat:] = process_dice(imp.local_importance[0])
        
      elif parser.exp_type == 'gams':
        ebm_local = ebm.explain_local(X_train[inds[j]:inds[j]+1], y_train[inds[j]:inds[j]+1])
        input_space[i+k][j][-num_feat:] = ebm_local._internal_obj['specific'][0]['scores'][:num_feat]
  
        
    if has_bug: #trying this instead of if sum is greater than 0
      has_bugs[i+k] = 1

    # if sum(which_has_bugs[inds]) > 0:
    #   has_bugs[i+k] = 1

np.save("../all_datasets/"+parser.dataset+"_dataset/dataset_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", input_space)
np.save("../all_datasets/"+parser.dataset+"_dataset/bugs_"+parser.exp_type+"_"+str(n)+"_"+parser.bug_type+"_"+str(parser.set_size)+".npy", has_bugs)

