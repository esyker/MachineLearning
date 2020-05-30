# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:00:30 2019

@author: Diogo
"""

import numpy as np
import matplotlib
import sklearn.linear_model as lm


X = np.load('data3_x.npy')
y = np.load('data3_y.npy')

#vetor de alphas
a = np.linspace(0,10,1000)
beta_Ridge = np.zeros((1000,3))

#calculo dos betas de Ridge para cada valor de alpha
for i in range(1000):
    ridge = lm.Ridge(alpha = a[i], max_iter = 10000, copy_X = 1)
    ridge.fit(X,y)
    beta_Ridge[i] = ridge.coef_
    
#plot dos valores beta em funçao de alpha para a Ridge regression
matplotlib.pyplot.figure()
"""matplotlib.pyplot.xscale("logit")"""
matplotlib.pyplot.plot(a,beta_Ridge[0]*np.ones((1000,3)))
matplotlib.pyplot.plot(a,beta_Ridge)
matplotlib.pyplot.legend(('b1 (alpha = 0)','b2 (alpha = 0)','b3 (alpha = 0)','b1','b2','b3'))
    
beta_Lasso = np.zeros((1000,3))

for i in range(1000):
    lasso = lm.Lasso(alpha = a[i], max_iter = 10000, copy_X = 1)
    lasso.fit(X,y)
    beta_Lasso[i] = lasso.coef_
    
#plot dos valores beta em funçao de alpha para Lasso
matplotlib.pyplot.figure()
#matplotlib.pyplot.xscale("logit")
matplotlib.pyplot.plot(a,beta_Lasso[0]*np.ones((1000,3)))
matplotlib.pyplot.plot(a,beta_Lasso)
matplotlib.pyplot.legend(('b1 (alpha = 0)','b2 (alpha = 0)','b3 (alpha = 0)','b1','b2','b3'))

a_ = 0.5
lasso = lm.Lasso(alpha = a_, max_iter = 10000, copy_X = 1)
lasso.fit(X,y)
SSE_lasso = 0

for i in range(50):
    SSE_lasso = SSE_lasso + (y[i] - lasso.predict(X)[i])**2
    
indices = np.ones(50);
for i in range(50):
    indices[i] = indices[i]*(i+1)
    


linear = lm.LinearRegression(copy_X = 1)
linear.fit(X,y)
SSE_linear = 0
for i in range(50):
    SSE_linear = SSE_linear + (y[i] - linear.predict(X)[i])**2
    
#plot dos valores de y para Lasso e LS
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(indices, lasso.predict(X))
matplotlib.pyplot.plot(indices, linear.predict(X))
matplotlib.pyplot.legend(('lasso','LS'))
    
print("SSE linear:", SSE_linear)
print("SSE Lasso:" ,SSE_lasso)
