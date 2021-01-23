# -*- coding: utf-8 -*-
"""
This script shows how to apply k-folds train and validate regression model to predict
MOS from the features computed with compute_features_example.m

Author: Zhengzhong Tu
"""
import warnings
import time
# Load libraries
import pandas
import random as rnd
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.stats
import scipy.io
from scipy.optimize import curve_fit
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
# ignore all warnings
warnings.filterwarnings("ignore")
# ===========================================================================
# Here starts the main part of the script
#
'''======================== parameters ================================''' 

model_name = 'XGB' # {SVR, RFR, XGB, SGD, GBM}
algo_name = 'RAPIQUE-3884'
data_name_train = 'KONVID_1K'
content = 'all' # options: {'screen', 'animation', 'gaming', 'natural'}
data_name_test = 'LIVE_VQC'
color_only = False

# load training feature mat and score
csv_file_train = './mos_files/'+data_name_train+'_metadata.csv'
mat_file_train = './feat_files_old/'+data_name_train+'_'+algo_name+'_feats.mat'
try:
    df = pandas.read_csv(csv_file_train, skiprows=[], header=None)       
except:
    raise Exception('Read csv file error!')
array = df.values
if data_name_train == 'LIVE_VQC' or data_name_train == 'KONVID_1K':
    y_train = array[1:,1] # for LIVE and KonVid
elif data_name_train == 'YOUTUBE_UGC' or data_name_train == 'YOUTUBE_UGC_natural':
    y_train = array[1:,4]
y_train = np.array(list(y_train), dtype=np.float)
if data_name_train == 'LIVE_VQC':
    y_train = np.divide(y_train, 100.0) * 4.0 + 1.0
X_mat_train = scipy.io.loadmat(mat_file_train)
X_train = np.asarray(X_mat_train['feats_mat'], dtype=np.float)
X_train[np.isnan(X_train)] = 0
X_train[np.isinf(X_train)] = 0

# load testing feature mat and score
csv_file_test = './mos_files/'+data_name_test+'_metadata.csv'
mat_file_test = './feat_files_old/'+data_name_test+'_'+algo_name+'_feats.mat'
try:
    df = pandas.read_csv(csv_file_test, skiprows=[], header=None)       
except:
    raise Exception('Read csv file error!')
array = df.values  
if data_name_test == 'LIVE_VQC' or data_name_test == 'KONVID_1K':
    y_test = array[1:,1] # for LIVE and KonVid
elif data_name_test == 'YOUTUBE_UGC' or data_name_test == 'YOUTUBE_UGC_natural':
    y_test = array[1:,4]
y_test = np.array(list(y_test), dtype=np.float)
if data_name_test == 'LIVE_VQC':
    y_test = np.divide(y_test, 100.0) * 4.0 + 1.0
X_mat_test = scipy.io.loadmat(mat_file_test)
X_test = np.asarray(X_mat_test['feats_mat'], dtype=np.float)
X_test[np.isnan(X_test)] = 0
X_test[np.isinf(X_test)] = 0

# define functions
import time
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from scipy.optimize import curve_fit
import scipy.stats

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat

def compute_metrics(y_pred, y):
  '''
  compute metrics btw predictions & labels
  '''
  # compute SRCC & KRCC
  SRCC = scipy.stats.spearmanr(y, y_pred)[0]
  try:
    KRCC = scipy.stats.kendalltau(y, y_pred)[0]
  except:
    KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

  # logistic regression btw y_pred & y
  beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
  popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_pred, *popt)
  
  # compute  PLCC RMSE
  PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
  RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
  return [SRCC, KRCC, PLCC, RMSE]

def formatted_print(snapshot, params, duration):
  print('======================================================')
  print('params: ', params)
  print('SRCC_train: ', snapshot[0])
  print('KRCC_train: ', snapshot[1])
  print('PLCC_train: ', snapshot[2])
  print('RMSE_train: ', snapshot[3])
  print('======================================================')
  print('SRCC_test: ', snapshot[4])
  print('KRCC_test: ', snapshot[5])
  print('PLCC_test: ', snapshot[6])
  print('RMSE_test: ', snapshot[7])
  print('======================================================')
  print(' -- ' + str(duration) + ' seconds elapsed...\n\n')

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# #############################################################################
# Train
#
# split into 5 folds
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train = scaler.transform(X_train)

if model_name == 'SVR':
    param_grid = {'C': np.logspace(1, 10, 10, base=2),
                  'gamma': np.logspace(-8, 1, 10, base=2)}
    grid = RandomizedSearchCV(SVR(kernel='rbf'), param_grid, cv=5, n_jobs=4)
elif model_name == 'RFR':
    param_grid = {'n_estimators': [100, 200, 300, 400, 500],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [3, 4, 5, 6, 7, 9, 11, -1],
                'min_samples_split': [2, 5, 10, 15],
                'min_samples_leaf': [1, 2, 5],
                'bootstrap': [True, False]}
    grid = RandomizedSearchCV(RandomForestRegressor(), param_grid, cv=5)
elif model_name == 'SGD':
    param_grid = {'alpha': 10.0**-np.arange(-1, 7), 
                  'penalty':["elasticnet", "l1", "l2"]}
    grid = RandomizedSearchCV(SGDRegressor(loss='epsilon_insensitive', average=True),
                              param_grid, cv=5)
elif model_name == 'XGB':
    param_grid = {'max_depth': range(3,12),
                'min_child_weight': range(1,10),
                'gamma': list([i/10.0 for i in range(0,5)]),
                'subsample': list([i/10.0 for i in range(6,10)]),
                'colsample_bytree': list([i/10.0 for i in range(6,10)]),
                'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
    grid = RandomizedSearchCV(XGBRegressor(objective ='reg:squarederror'), param_grid, cv=5)
elif model_name == 'GBM':
    param_grid = {'num_leaves': [7, 15, 31, 61, 81, 127],
                    'max_depth': [3, 4, 5, 6, 7, 9, 11, -1],
                   'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
                   'n_estimators': [100, 200, 300, 400, 500],
                   'boosting_type': ['gbdt', 'dart'],
                   'class_weight': [None, 'balanced'],
                   'min_child_samples': [10, 20, 40, 60, 80, 100, 200],
                   'bagging_freq': [0, 3, 9, 11, 15, 17, 23, 31],
                   'subsample': [0.5, 0.7, 0.8, 0.9, 1.0],
                   'reg_alpha':[1e-5, 1e-2, 0.1, 1, 10, 100],
                   'reg_lambda': [1e-5, 1e-2, 0.1, 1, 10, 100],
                   'objective': [None, 'mse', 'mae', 'huber'],
                   }
    grid = RandomizedSearchCV(LGBMRegressor(), param_grid, cv=5)
t_start = time.time()

# grid search
#grid.fit(X_train, y_train)
#best_params = grid.best_params_
# init model
if model_name =='SGD':
    regressor = SGDRegressor(loss='epsilon_insensitive', average=True, **best_params)
elif model_name == 'SVR':
   #  regressor = SVR(kernel='rbf',**best_params)
    regressor = SVR()	
elif model_name == 'RFR':
   # regressor = RFR(**best_params)
    regressor = RandomForestRegressor(n_estimators=100)
elif model_name == 'XGB':
    # regressor = XGBRegressor(objective ='reg:squarederror', **best_params)
    regressor = XGBRegressor(objective ='reg:squarederror')
elif model_name == 'GBM':
    regressor = LGBMRegressor(**best_params)
# re-train the model using the best alpha
regressor.fit(X_train, y_train)

# predictions
y_train_pred = regressor.predict(X_train)
X_test = scaler.transform(X_test)
y_test_pred = regressor.predict(X_test)

# compute metrics
metrics_train = compute_metrics(y_train_pred, y_train)
metrics_test = compute_metrics(y_test_pred, y_test)

# print values
t_end = time.time()
formatted_print(metrics_train + metrics_test, regressor.get_params(), (t_end - t_start))
