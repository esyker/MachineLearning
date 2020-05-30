# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:29:58 2019

@author: diogo
"""

import numpy as np
import sklearn.metrics as skmetrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sklearn.naive_bayes as nb
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection as ms
from sklearn.metrics import balanced_accuracy_score

import warnings

warnings.filterwarnings('ignore', 'Solver terminated early.*')

for dataset_number in range(1,3):
    if dataset_number == 1:
        print("\ndataset1:")
        dataset_xtrain=np.load('dataset1_xtrain.npy')
        dataset_ytrain=np.load('dataset1_ytrain.npy')
        dataset_xtest=np.load('dataset1_xtest.npy')
        dataset_ytest=np.load('dataset1_ytest.npy')
    elif dataset_number == 2:
        print("\ndataset2:")
        dataset_xtrain=np.load('dataset2_xtrain.npy')
        dataset_ytrain=np.load('dataset2_ytrain.npy')
        dataset_xtest=np.load('dataset2_xtest.npy')
        dataset_ytest=np.load('dataset2_ytest.npy')
    
    X_train, X_validation, y_train, y_validation = ms.train_test_split(dataset_xtrain, dataset_ytrain, test_size=0.3)
    
    num_tests = 100
    y_predict=np.zeros((y_validation.size,1))
    scores=np.zeros(num_tests)
    c0=0
    for j in range(num_tests):
        #fit the SVM to the data with different polynomial orders j
        clf = SVC(kernel='poly',max_iter=100000,degree=j,coef0=c0,gamma='auto')
        clf.fit(X_train,np.ravel(y_train))
        y_predict=clf.predict(X_validation)
        scores[j]=accuracy_score(y_validation,y_predict)
        clf=None
    max_degree = np.argmax(scores)
    clf = SVC(kernel='poly',max_iter=100000,degree=max_degree,coef0=c0,gamma='auto')
    clf.fit(dataset_xtrain,np.ravel(dataset_ytrain))
    y_predict=clf.predict(dataset_xtest)
    score=accuracy_score(dataset_ytest,y_predict)
    balanced_score=balanced_accuracy_score(dataset_ytest,y_predict)
    confusion_matrix = skmetrics.confusion_matrix(dataset_ytest,y_predict)
    print("SVM Polynomial with degree",max_degree,"\t Score: ",score,"\t Balanced Score: ",balanced_score, "\t Confusion Matrix: ",confusion_matrix[0], confusion_matrix[1])
    clf = None
    
    gammas=np.logspace(start=-8,stop=1,num=num_tests)#create various values of gamma to test
    for j in range(num_tests):
        #fit the SVM to the data with different polynomial orders j
        clf = SVC(kernel='rbf',max_iter=100000,gamma=gammas[j])
        clf.fit(X_train,np.ravel(y_train))
        y_predict=clf.predict(X_validation)
        scores[j]=accuracy_score(y_validation,y_predict)
        clf=None
    max_gamma_index = np.argmax(scores)
    clf = SVC(kernel='rbf',max_iter=100000,gamma=gammas[max_gamma_index])
    clf.fit(dataset_xtrain,np.ravel(dataset_ytrain))
    y_predict=clf.predict(dataset_xtest)
    score=accuracy_score(dataset_ytest,y_predict)
    balanced_score=balanced_accuracy_score(dataset_ytest,y_predict)
    confusion_matrix = skmetrics.confusion_matrix(dataset_ytest,y_predict)
    print("SVM RBF with gamma",gammas[max_gamma_index],"\t Score: ",score,"\t Balanced Score: ",balanced_score, "\t Confusion Matrix: ",confusion_matrix[0], confusion_matrix[1])
    clf = None
    
    for j in range(1,num_tests):
        #fit the SVM to the data with different polynomial orders j
        clf = KNeighborsClassifier(j)
        clf.fit(X_train,np.ravel(y_train))
        y_predict=clf.predict(X_validation)
        scores[j]=accuracy_score(y_validation,y_predict)
        clf=None
    max_neighbours = np.argmax(scores)
    clf = KNeighborsClassifier(max_neighbours)
    clf.fit(dataset_xtrain,np.ravel(dataset_ytrain))
    y_predict=clf.predict(dataset_xtest)
    score=accuracy_score(dataset_ytest,y_predict)
    balanced_score=balanced_accuracy_score(dataset_ytest,y_predict)
    confusion_matrix = skmetrics.confusion_matrix(dataset_ytest,y_predict)
    print("Nearest Neighbour with neighbours ",max_neighbours,"\t Score: ",score,"\t Balanced Score: ",balanced_score, "\t Confusion Matrix: ",confusion_matrix[0], confusion_matrix[1])
    clf = None
    
    
    clf = nb.MultinomialNB(alpha = 1, fit_prior = False)
    clf.fit(dataset_xtrain,np.ravel(dataset_ytrain))
    y_predict=clf.predict(dataset_xtest)
    score=accuracy_score(dataset_ytest,y_predict)
    balanced_score=balanced_accuracy_score(dataset_ytest,y_predict)
    confusion_matrix = skmetrics.confusion_matrix(dataset_ytest,y_predict)
    print("Multinomial Naive Bayes \t Score: ",score,"\t Balanced Score: ",balanced_score, "\t Confusion Matrix: ",confusion_matrix[0], confusion_matrix[1])
    clf=None
    
    clf = nb.GaussianNB()
    clf.fit(dataset_xtrain,np.ravel(dataset_ytrain))
    y_predict=clf.predict(dataset_xtest)
    score=accuracy_score(dataset_ytest,y_predict)
    balanced_score=balanced_accuracy_score(dataset_ytest,y_predict)
    confusion_matrix = skmetrics.confusion_matrix(dataset_ytest,y_predict)
    print("Gaussian Naive Bayes \t Score: ",score,"\t Balanced Score: ",balanced_score, "\t Confusion Matrix: ",confusion_matrix[0], confusion_matrix[1])
    clf=None
    
    clf = tree.DecisionTreeClassifier()
    clf.fit(dataset_xtrain,np.ravel(dataset_ytrain))
    y_predict=clf.predict(dataset_xtest)
    score=accuracy_score(dataset_ytest,y_predict)
    balanced_score=balanced_accuracy_score(dataset_ytest,y_predict)
    confusion_matrix = skmetrics.confusion_matrix(dataset_ytest,y_predict)
    print("Decision Tree \t Score: ",score,"\t Balanced Score: ",balanced_score, "\t Confusion Matrix: ",confusion_matrix[0], confusion_matrix[1])
    clf = None
