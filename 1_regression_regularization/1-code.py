# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib




x_training = np.load('data2a_x.npy')
y_training = np.load('data2a_y.npy')

N = x_training.size
P = 2

x_plot = np.linspace(-1,1,100) # 100 linearly spaced numbers

X = np.zeros((N,P+1))

#calculo da matrix X
for i in range(N):
    for j in range(P+1):
        if j == 0:
            X[i][j] = 1
        else:
            X[i][j] = x_training[i]**j


#cálculo da estimativa de beta
beta = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),y_training))

#função que estima o valor de y com base nos coeficientes obtidos pelo método de LS
def y_estimate(x):
    for j in range(P+1):
        if j == 0:
            y = beta[0]
        else:
            y = y + beta[j]*(x**j)
    return y
    
#calculo do SSE
SSE = 0
for i in range(N):
    SSE = SSE + (y_training[i]-y_estimate(x_training[i]))**2
            
#plot de y            
y_plot = np.zeros(100)
for i in range(100):
    y_plot[i] = y_estimate(x_plot[i])

matplotlib.pyplot.scatter(x_training, y_training)
matplotlib.pyplot.plot(x_plot,y_plot)

print("SSE:" ,SSE)
print("BETA:" , beta)
