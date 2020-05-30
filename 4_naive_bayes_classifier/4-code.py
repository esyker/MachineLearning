# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:14:38 2019

@author: Diogo
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

#2.1. Load the numpy files for data1 which have already been split into training 
#data (variables xtrain and ytrain) and test data (variables xtest and ytest).
xtest = np.load('data1_xtest.npy')
xtrain = np.load('data1_xtrain.npy')
ytest = np.load('data1_ytest.npy')
ytrain = np.load('data1_ytrain.npy')

#2.2. Obtain a scatter plot of the training and test data, using different colors, or symbols,
#for the different classes. Don’t forget to use equal scales for both axes, so that the
#scatter plot is not distorted.

#colors for each class
c1 = 'orange' 
c2 = 'green'
c3 = 'blue'

plt.figure()
plt.axis('equal') #same scale for both axis

#train arrays to separate the data into each class
xtrain_class1 = np.empty((0,2))
xtrain_class2 = np.empty((0,2))
xtrain_class3 = np.empty((0,2))
label1 = 'Class 1'
label2 = 'Class 2'
label3 = 'Class 3'
size = 10
#plot/build arrays for training set
for i in range(ytrain.size):
    if ytrain[i] == 1:
        plt.scatter(xtrain[i][0],xtrain[i][1], c=c1, s = size, label=label1) #plot for class 1
        xtrain_class1 = np.vstack((xtrain_class1, xtrain[i])) #build array for class 1
        label1 = ''
    if ytrain[i] == 2:
        plt.scatter(xtrain[i][0],xtrain[i][1], c=c2, s = size, label=label2)
        xtrain_class2 = np.vstack((xtrain_class2, xtrain[i]))
        label2 = ''
    if ytrain[i] == 3:
        plt.scatter(xtrain[i][0],xtrain[i][1], c=c3, s = size, label=label3)
        xtrain_class3 = np.vstack((xtrain_class3, xtrain[i]))
        label3 = ''
        
#plot test dataset
for i in range(ytest.size):
    if ytest[i] == 1:
        plt.scatter(xtest[i][0],xtest[i][1], s = size, c=c1) 
    if ytest[i] == 2:
        plt.scatter(xtest[i][0],xtest[i][1], s = size, c=c2)
    if ytest[i] == 3:
        plt.scatter(xtest[i][0],xtest[i][1], s = size, c=c3)
    
plt.legend()
plt.title('Scatter plot of training and test data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


#2.3. Make a script that creates a naive Bayes classifier based on the training data, and that
#finds the classifications that that classifier gives to the test data. The script should plot
#the classifications of the test data as a function of the test pattern index, and should
#print the percentage of errors that the classifier makes on the test data.

#calculate mean and var of x for each class
xtrain_mean_class1 = np.mean(xtrain_class1, axis = 0)
xtrain_var_class1 = np.var(xtrain_class1, axis = 0, ddof = 0) #ddof = 0 -> divide by N and not N-1
xtrain_mean_class2 = np.mean(xtrain_class2, axis = 0)
xtrain_var_class2 = np.var(xtrain_class2, axis = 0, ddof = 0)
xtrain_mean_class3 = np.mean(xtrain_class3, axis = 0)
xtrain_var_class3 = np.var(xtrain_class3, axis = 0, ddof = 0)

#calculate covar matrix (we assume it is a diagonal matrix)
xtrain_covar_class1 = np.identity(2) * xtrain_var_class1
xtrain_covar_class2 = np.identity(2) * xtrain_var_class2
xtrain_covar_class3 = np.identity(2) * xtrain_var_class3

#calculate the "probability" of each point in the test setbelonging to each of the 3 classes
p_class1 = ss.multivariate_normal.pdf(xtest, xtrain_mean_class1, xtrain_covar_class1)
p_class2 = ss.multivariate_normal.pdf(xtest, xtrain_mean_class2, xtrain_covar_class2)
p_class3 = ss.multivariate_normal.pdf(xtest, xtrain_mean_class3, xtrain_covar_class3)

#the predicted class will be the one in which the probability is bigger
y_predict = np.transpose(np.asmatrix(np.argmax(np.array([p_class1, p_class2, p_class3]), axis = 0) + 1)) # the +1 is for converting index to class (0,1,2 -> 1,2,3)

#2.4. Plot the classifications of the test data as a function of the test pattern index.
plt.figure()
for i in range(ytest.size):
    plt.scatter(i,ytest[i], c='blue', s= 50, label='y_test' if i == 0 else '')
for i in range(y_predict.size):
    plt.scatter(i, float(y_predict[i]), c='orange', s = 10, label='y_predict' if i == 0 else '')
plt.legend()
plt.title('Classifications of the test data as a function of the test pattern index')
plt.ylabel('Class')
plt.xlabel('Index')

#2.5. Indicate the percentage of errors that you obtained in the test set.
errors = 0
for i in range(ytest.size):
    if y_predict[i] != ytest[i]:
        errors = errors + 1

percentage_errors = errors/ytest.size

print("Percentage of errors: ", percentage_errors*100, "%")



