#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:42:46 2024

@author: ozaballa
"""


from model import *
import numpy as np
from scipy.io import loadmat  
from scipy.io import savemat  
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import csv
import pandas as pd
import os
import seaborn as sns



np.set_printoptions(threshold=np.inf)
# data in .mat file
path = 'data/'
filename = 'gefcom2017'
mat = loadmat(path + filename + '.mat')  
mdata = mat['data'] 
mdtype = mdata.dtype 
data = {n: mdata[n][0, 0] for n in mdtype.names}


dataset = range(0,data['c'].shape[1])


for key, matrix in data.items():
    column = np.array([row[dataset] for row in matrix])
    data[key] = column
    


lambda_s = 0.9 
lambda_r = 0.9 
print('lambda_s ' + str(lambda_s) + ', lambda_r ' + str(lambda_r) )


n_years = 1
days_train = 30 
n_train = 24*days_train
L = 24 # prediction horizon (hours)
C = 48 # length of the calendar information
R = 3 # length of feature vector of observations
n = len(data.get('consumption'))
consumption = data.get('consumption').T
K  = consumption.shape[0]
ct = data.get('c')
temperature = data.get('temperature') 
hour_of_prediction = 10 # last observed hour before starting the prediction
                     

initial_date = data['date'][0,0] + 24*3600+ 3600*(hour_of_prediction- data['c'][0,0]) 
start_index = np.where(data['date']==initial_date)[0][0]
n = np.where( (data['DoY'][:,0]==31) & (data['month'][:,0]==12) & (data['HoD'][:,0]==hour_of_prediction- 1))[0][n_years-1] # last day of prediction


[Theta, Gamma] = initialize(C, R, K)    
predictions = []
estimated_errors = list()
load_demand = []


# Learning step:
for i in range(start_index, n_train+L, L):
    s0 = consumption[:,i:i+1]
    w = temperature[i+1:i+L+1,:].T 
    x = [s0, w] 
    y = consumption[:,i+1:i+L+1] # real load demand of the next 24h
    c = ct[i+1:i+L+1] # calendar for the next 24h 
    c = c.astype(int)

    # Aprendemos ST:
    for k in range(K):
        y_k = y[k,:][np.newaxis,:]
        s0_k = s0[k,np.newaxis]
        w_k = w[k, np.newaxis]
        x_k = [s0_k, w_k]
        c_k = c[:,k][:,np.newaxis]     
    [Theta, Gamma] = update_model(Theta, Gamma, y, x, c, lambda_s, lambda_r) 

# Prediction step
prediction_calendar = []
j=i+L
print('Hour of prediction ' + str(ct[j+1,0] ))


for j in range(i+L, n, L):
    s0 = consumption[:,j:j+1] # K
    w = temperature[j+1:j+L+1].T # 24 x K
    x = [s0, w]
    c = ct[j+1:j+L+1]
    c = c.astype(int)
    prediction_calendar.append(c)
    cont =+1

    Theta_pred = adapt_covariance(Theta, K, C) 
    # Theta_pred = Theta
    [pred_s, e] = prediction(Theta_pred, x, c, K) 
    if np.sum(pred_s<0)>0 :
        pred_s[np.where(pred_s<0)] = 0
    predictions = np.append(predictions, pred_s)
    for cal in e:
        estimated_errors.append(cal) 
    y = consumption[:,j+1:j+L+1]
    load_demand = np.append(load_demand, np.transpose(y))

    [Theta, Gamma] = update_model(Theta, Gamma, y, x, c, lambda_s, lambda_r) 


[MAPE, RMSE, MAPE_K, RMSE_K] = test(predictions, load_demand, K)
print('MAPE = ' + str(MAPE))
print('RMSE = ' + str(RMSE))


predictions_K = np.array(predictions).reshape(K, int(len(predictions)/K), order = 'F') #Results per task
load_demand_K = np.array(load_demand).reshape(K, int(len(load_demand)/K), order = 'F') 



results = {}
results['lambda_s'] = lambda_s
results['lambda_r'] = lambda_r
results['ntrain'] = n_train
results['H'] = n
results['t0'] = str(start_index + n_train + 1)
results['ytest'] = load_demand_K
results['ypred'] = predictions_K
results['ypred_var'] = estimated_errors
results['RMSE'] = RMSE
results['MAPE'] = MAPE
results['RMSE_K'] = RMSE_K
results['MAPE_K'] = MAPE_K



# results['data'] = data
 
savemat(filename + '_results.mat', results)


