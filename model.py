#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 08:55:17 2024

@author: ozaballa
"""

import numpy as np
import copy


def initialize(C_init, R_init, K_init):
  # initialize parameters

  global R, C, K
  R = R_init
  C = C_init
  K = K_init
  
  import numpy as np
  class Theta:
    pass
  class Gamma:
    pass
  a = Theta()
  a.eta_s = np.zeros((C, K, K+1))
  a.sigma_s = np.zeros((C, K, K))
  a.eta_r = np.zeros((C, K, K*R))
  a.sigma_r = np.zeros((C, K, K))
  a.w_t = np.zeros((K, C)) 
  a.sigma_t = np.zeros((C, K))  
  
  b = Gamma()
  b.gamma_t = np.zeros((C, K)) 
  b.P_t = np.ones((C, K, 1))
  b.gamma_s = np.zeros((C, 1)) 
  b.gamma_r = np.zeros((C, 1))
  b.P_s = np.zeros((C, K+1, K+1))
  b.P_r = np.zeros((C, K*R, K*R))
  for i in range(C):
      b.P_s[i] = np.eye(K+1)
      b.P_r[i] = np.eye((K*R))
  return a, b



def update_parameters(eta, sigma, P, gamma, l, s, u):
    if np.sum(P.trace()) > 10**10:
        P = np.eye(len(P))
    P =  (1/l)*(P - np.dot(np.dot(  np.dot(P, u), 1/(l + np.dot(np.dot(u.T, P),u)) ) , np.dot(u.T, P) )) 
    gamma = l*gamma + 1
    K = s.shape[0]
    sigma = sigma - (1/gamma)*(sigma - (l/(l+ np.dot(np.dot(u.T,P),u)))**2 * np.dot((s - np.dot(eta,u)), (s.T - np.dot(u.T,eta.T )))) 
    eta = eta + (1/(l + np.dot(np.dot(u.T,P.T),u)) ) * np.dot((s - np.dot(eta,u)), np.dot(u.T,P.T))

    return eta, sigma, P, gamma

def update_ur(eta, P, l, s, u): 
    if P.shape[0] > 1: # if K > 1
      if np.sum(P) > 10**2:
        P = np.ones((P.shape[0], 1))
    else:  
      if P>10**1:
        P=1
    P = (1/l)*(P - P* u* (1/(l + u*P*u))*u* P)
    eta = eta + (s - eta*u)* (1/(l + u*P*u))*u*P
    return eta, P



def update_model(Theta, Gamma, y, x, c, lambda_s, lambda_r): 
  s0 = x[0] # load
  K = len(s0)
  w = x[1] # temperature 
  L = len(y[0])
  y = np.hstack((s0, y[0:]))
  
  
  
  for j in range(L): 
    u_r = np.ones((K*R,1))
    [Theta.w_t[:,c[j][0]:c[j][0]+1], Gamma.P_t[c[j][0]]] = update_ur(Theta.w_t[:,c[j][0]:c[j][0]+1], Gamma.P_t[c[j][0]], 1, w[:,j][:, np.newaxis], np.ones((K,1)))
    
    for k in range(K):
        if Theta.w_t[k,c[j][0]] - w[k,j] > 20 and (w[k,j] > 80 or w[k,j] < 20): 
            alpha1 = 1
            alpha2 = 0
        elif Theta.w_t[k,c[j][0]] - w[k,j] < -20 and (w[k,j] > 80 or w[k,j] < 20):
            alpha1 = 0
            alpha2 = 1
        else:
            alpha1 = 0
            alpha2 = 0
        u_r[3*k + 1, 0] = alpha1
        u_r[3*k + 2, 0] = alpha2
    u_s = np.ones((K+1,1)) 
    u_s[1:, :] = y[:,j:j+1]
    
    # Update of the parameters for each c
    [Theta.eta_s[c[j][0]], Theta.sigma_s[c[j][0]], Gamma.P_s[c[j][0]], Gamma.gamma_s[c[j][0]]] = update_parameters(Theta.eta_s[c[j][0]], Theta.sigma_s[c[j][0]], Gamma.P_s[c[j][0]], Gamma.gamma_s[c[j][0]], lambda_s, y[:,j+1:j+2], u_s)
    [Theta.eta_r[c[j][0],:], Theta.sigma_r[c[j][0]], Gamma.P_r[c[j][0]], Gamma.gamma_r[c[j][0]]] = update_parameters(Theta.eta_r[c[j][0],:], Theta.sigma_r[c[j][0]], Gamma.P_r[c[j][0]], Gamma.gamma_r[c[j][0]], lambda_r, y[:,j+1:j+2], u_r) 
  

  return Theta, Gamma



def prediction(theta, x, c, K):
  L = x[1].shape[1]
  pred_s = np.zeros((L+1, K, 1)) 
  e = np.zeros((L+1, K, K)) 
  pred_s[0,:] = x[0]
  w = x[1]
  i=0
  for i in range(L):
    # print(i)
    u_s = np.insert(pred_s[i], 0, 1)[:,np.newaxis]
    N = np.vstack((np.zeros((1,K)),np.eye(K)))
    u_r = np.ones((K*R,1))
    for k in range(K):
        if theta.w_t[k,c[i][0]] - w[k,i] > 20 and (w[k,i] > 80 or w[k,i] < 20): 
            alpha1 = 1
            alpha2 = 0
        elif theta.w_t[k,c[i][0]] - w[k,i] < -20 and (w[k,i] > 80 or w[k,i] < 20):
            alpha1 = 0
            alpha2 = 1
        else:
            alpha1 = 0
            alpha2 = 0
        u_r[3*k + 1, 0] = alpha1
        u_r[3*k + 2, 0] = alpha2
     
    W1 = theta.sigma_s[c[i]][0] + np.dot(np.dot( np.dot(theta.eta_s[c[i]][0], N), e[i]) ,  np.dot(theta.eta_s[c[i]][0], N).T )
    W2 = theta.sigma_r[c[i]][0]
    inverseW = np.linalg.inv(W1 + W2)
    
    pred_s[i+1] = np.dot(np.dot(W1, inverseW ) , np.dot(theta.eta_r[c[i]][0], u_r) ) +  np.dot(np.dot(W2, inverseW ) , np.dot(theta.eta_s[c[i]][0], u_s) )
    e[i+1] = np.dot(np.dot(W2 ,inverseW) , W1)        

  return pred_s[1:], e[1:]



def test(predictions, load_demand, K):
    m = np.abs( (predictions - load_demand)/load_demand) 
    r = (predictions - load_demand)**2
    m_K = np.array(m).reshape(K, int(len(m)/K), order = "F" )
    r_K = np.array(r).reshape(K, int(len(r)/K), order = "F" )
    MAPE_K = 100*np.nanmean(m_K, axis = 1) 
    RMSE_K = np.sqrt(np.nanmean(r_K, axis = 1)) 
    MAPE = 100*np.nanmean(m_K) 
    RMSE = np.nanmean(RMSE_K) 
    return MAPE, RMSE, MAPE_K, RMSE_K


def adapt_covariance(Theta_multitask, K, C):
    Theta_1 = copy.deepcopy(Theta_multitask)
    for c in range(C):

        cov_matrix = Theta_1.sigma_s[c]
        variances = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(variances, variances)
        
        mask = (np.abs(corr_matrix) < 0.1 ) # los que quiero que SI se hagan 0
        matrix_aux = Theta_1.sigma_s[c]*(mask)
        
        row_sums = np.sum(np.abs(matrix_aux), axis=1)
        matrix_aux = matrix_aux*(-1)
        matrix_aux[ np.eye(K, dtype=bool)] = matrix_aux[ np.eye(K, dtype=bool)] + row_sums
        
        Theta_1.sigma_s[c] = Theta_1.sigma_s[c] + matrix_aux


        cov_matrix = Theta_1.sigma_r[c]
        variances = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(variances, variances)

        mask = (np.abs(corr_matrix) < 0.1 ) # los que quiero que SI se hagan 0
        matrix_aux = Theta_1.sigma_r[c]*(mask)
        
        row_sums = np.sum(np.abs(matrix_aux), axis=1)
        matrix_aux = matrix_aux*(-1)
        matrix_aux[ np.eye(K, dtype=bool)] = matrix_aux[ np.eye(K, dtype=bool)] + row_sums
        
        Theta_1.sigma_r[c] = Theta_1.sigma_r[c] + matrix_aux

    return Theta_1