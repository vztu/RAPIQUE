# -*- coding: utf-8 -*-
"""
This script shows how to apply 80-20 holdout train and validate regression model to predict
MOS from the features computed with compute_features_example.m

Author: Zhengzhong Tu
"""
# Load libraries
# import warnings
import os 
import pandas
import scipy.io
import numpy as np
# warnings.filterwarnings("ignore")

# ===========================================================================
# Here starts the main part of the script
#
'''======================== parameters ================================''' 

model_name = 'SVR'  # {SVR, RFR, XGB, SGD, GBM}
data_name = 'ALL_COMBINED'
color_only = False # if True, it is YouTube-UGCc dataset; if False it is YouTube-UGC
use_inlsa = True # if True, apply INLSA calibration of MOS.
algo_name = 'RAPIQUE'
test_size = 0.2
# csv_file = './mos_files/'+data_name+'_metadata.csv'
# mat_file = './feat_files/'+data_name+'_'+algo_name+'_feats.mat'
corr_result_file = './corr_results/'+data_name+'_'+algo_name+'_corr.mat'

'''======================== read files =============================== '''
## read KONVID_1K
data_name = 'KONVID_1K'
csv_file = os.path.join('mos_files', data_name+'_metadata.csv')
mat_file = os.path.join('feat_files', data_name+'_'+algo_name+'_feats.mat')
df = pandas.read_csv(csv_file, skiprows=[], header=None)
array = df.values
y1 = array[1:,1]
y1 = np.array(list(y1), dtype=np.float)
X_mat = scipy.io.loadmat(mat_file)
X1 = np.asarray(X_mat['feats_mat'], dtype=np.float)
X1[np.isnan(X1)] = 0
X1[np.isinf(X1)] = 0
if use_inlsa:
    # apply scaling transform
    temp = np.divide(5.0 - y1, 4.0) # convert MOS do distortion
    temp = -0.0993 + 1.1241 * temp # apply gain and shift produced by INSLA
    temp = 5.0 - 4.0 * temp # convert distortion to MOS
    y1 = temp

## read LIVE-VQC
data_name = 'LIVE_VQC'
csv_file = os.path.join('mos_files', data_name+'_metadata.csv')
mat_file = os.path.join('feat_files', data_name+'_'+algo_name+'_feats.mat')
df = pandas.read_csv(csv_file, skiprows=[], header=None)
array = df.values  
y2 = array[1:,1]
y2 = np.array(list(y2), dtype=np.float)
X_mat = scipy.io.loadmat(mat_file)
X2 = np.asarray(X_mat['feats_mat'], dtype=np.float)
X2[np.isnan(X2)] = 0
X2[np.isinf(X2)] = 0

if use_inlsa:
    # apply scaling transform
    temp = np.divide(100.0 - y2, 100.0) # convert MOS do distortion
    temp = 0.0253 + 0.7132 * temp # apply gain and shift produced by INSLA
    temp = 5.0 - 4.0 * temp # convert distortion to MOS
    y2 = temp
else:
    # apply linear scaling
    y2 = np.divide(y2, 100.0) * 4.0 + 1.0

## read YOUTUBE_UGC
data_name = 'YOUTUBE_UGC'
csv_file = os.path.join('mos_files', data_name+'_metadata.csv')
mat_file = os.path.join('feat_files', data_name+'_'+algo_name+'_feats.mat')
df = pandas.read_csv(csv_file, skiprows=[], header=None)
array = df.values  
y3 = array[1:,4]
y3 = np.array(list(y3), dtype=np.float)
X_mat = scipy.io.loadmat(mat_file)
X3 = np.asarray(X_mat['feats_mat'], dtype=np.float)
X3[np.isnan(X3)] = 0
X3[np.isinf(X3)] = 0
#### 57 grayscale videos in YOUTUBE_UGC dataset, we do not consider them for fair comparison ####
if color_only:
    gray_indices = [3,6,10,22,23,46,51,52,68,74,77,99,103,122,136,141,158,173,368,426,467,477,506,563,594,\
    639,654,657,666,670,671,681,690,697,702,703,710,726,736,764,768,777,786,796,977,990,1012,\
    1015,1023,1091,1118,1205,1282,1312,1336,1344,1380]
    gray_indices = [idx - 1 for idx in gray_indices]
    X3 = np.delete(X3, gray_indices, axis=0)
    y3 = np.delete(y3, gray_indices, axis=0)
# concat three datasets
X = np.vstack((X1, X2, X3))
y = np.vstack((y1.reshape(-1,1), y2.reshape(-1,1), y3.reshape(-1,1))).squeeze()

X = X[:, list(range(0, 272))+list(range(680, 680+272))+list(range(1360,3884))]

## preprocessing
X[np.isinf(X)] = np.nan
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
X = imp.transform(X)
print(X.shape)

## define functions
import time
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

def final_avg(snapshot):
  def formatted(args, pos):
    median = np.median(list(map(lambda x: x[pos], snapshot)))
    stdev = np.std(list(map(lambda x: x[pos], snapshot)))
    print('{}: {} (std: {})'.format(args, median, stdev))

  print('======================================================')
  print('Average training results among all repeated 80-20 holdouts:')
  formatted("SRCC Train", 0)
  formatted("KRCC Train", 1)
  formatted("PLCC Train", 2)
  formatted("RMSE Train", 3)
  print('======================================================')
  print('Average testing results among all repeated 80-20 holdouts:')
  formatted("SRCC Test", 4)
  formatted("KRCC Test", 5)
  formatted("PLCC Test", 6)
  formatted("RMSE Test", 7)
  print('\n\n')


'''======================== parameters end ===========================''' 
print("Evaluating algorithm {} with {} on dataset {} ...".format(algo_name, model_name, data_name))


from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

all_iterations = []
log_verbose = True
t_overall_start = time.time()
# 100 times random train-test splits
for i in range(1,21):
  if log_verbose:
    print('{} th repeated holdout test'.format(i))
    t_start = time.time()

  # train test split
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_size, random_state=math.ceil(8.8*i))
  scaler = preprocessing.MinMaxScaler().fit(X_train)
  X_train = scaler.transform(X_train)
  # grid search CV on the training set
  # grid search CV on the training set
  if model_name == 'SVR':
    param_grid = {'C': np.logspace(1, 10, 10, base=2),
                  'gamma': np.logspace(-8, 1, 10, base=2)}
    grid = RandomizedSearchCV(SVR(), param_grid, cv=5, n_jobs=8)
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
                              param_grid, cv=3)
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
                    'max_depth': [3, 4, 5, 6, 7, 9, 11],
                   'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
                   'n_estimators': [100, 200, 300, 400, 500],
                   'boosting_type': ['gbdt', 'dart'],
                   'class_weight': [None, 'balanced'],
                   'min_child_samples': [10, 20, 40, 60, 80, 100, 200],
                #    'bagging_freq': [0, 3, 9, 11, 15, 17, 23, 31],
                   'subsample': [0.5, 0.7, 0.8, 0.9, 1.0],
                   'reg_alpha':[1e-5, 1e-2, 0.1, 1, 10, 100],
                   'reg_lambda': [1e-5, 1e-2, 0.1, 1, 10, 100],
                #    'objective': [None, 'mse', 'mae', 'huber'],
                   }
    grid = RandomizedSearchCV(LGBMRegressor(), param_grid, cv=5)
  # grid search
  grid.fit(X_train, y_train)
  best_params = grid.best_params_
  # init model
  if model_name =='SGD':
    regressor = SGDRegressor(loss='epsilon_insensitive', average=True, **best_params)
  elif model_name == 'SVR':
    regressor = SVR(**best_params)
  elif model_name == 'RFR':
    regressor = RFR(**best_params)
  elif model_name == 'XGB':
    regressor = XGBRegressor(objective ='reg:squarederror', **best_params)
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
  if log_verbose:
    t_end = time.time()
    formatted_print(metrics_train + metrics_test, best_params, (t_end - t_start))
  all_iterations.append(metrics_train + metrics_test)
  # save iters 
  pass

# formatted print overall iterations
final_avg(all_iterations)
print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))

#================================================================================
# save figures
scipy.io.savemat(corr_result_file, \
    mdict={'all_iterations': np.asarray(all_iterations,dtype=np.float)})
