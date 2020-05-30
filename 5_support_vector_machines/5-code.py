# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:29:19 2019

@author: diogo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math
import warnings

warnings.filterwarnings('ignore', 'Solver terminated early.*')


# =============================================================================
# Visualization Function
# =============================================================================
def plot_contours(clf, points,labels):
    """Plot the decision boundaries for a classifier.
    Parameters
    ----------
    clf: a SVM classifier
    params: dictionary of params to pass to contourf, optional
    """
    def make_meshgrid(x, y,h=0.02):
        """Create a mesh of points to plot in
        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional
        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
    X0, X1 = points[:, 0], points[:, 1]
    xx, yy = make_meshgrid(X0, X1,0.5)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #VISUALIZATION OF DECISION
    fig2 = plt.figure(figsize=(5,5))
    ax = fig2.add_subplot(1,1,1)
    out = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=80, edgecolors='k')
    ax.scatter(X0[clf.support_],X1[clf.support_],c=labels[clf.support_], cmap=plt.cm.coolwarm, s=80, edgecolors='w')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    plt.show()
    return out
# =============================================================================
# =============================================================================


#1(a) Load the files spiral_x.mat and spiral_y.mat which contain the classical spiral
#example, with 50 patterns per class.
spiral_x=np.load('spiral_X.npy')
spiral_y=np.load('spiral_Y.npy')


#Visualization of the input spiral data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(spiral_y.size):
    if(spiral_y[i]==1):
        ax.scatter(spiral_x[i][0],spiral_x[i][1],spiral_y[i], c='green',s=32,marker='o')
    else:
        ax.scatter(spiral_x[i][0],spiral_x[i][1],spiral_y[i], c='red',s=32,marker='^')
plt.show()
plt.title('1a) Visualization of the dataset')


#1(b) Instantiate an SVM classification model with a polynomial kernel. 
#Set the maximum number of iterations to 100000. Fit the SVM model to the spiral data.
#1(c) Determine experimentally, using the polynomial kernel, the value of p for which
#you get the best classifier (start with p = 1) on your training data. Write down all
#the experiments performed, together with the classification error percentages and
#number of support vectors.
print("\n3.1c)------------------------------------------------------")
num_tests = 100
#use SVM to classify data
classifications=np.zeros((num_tests,spiral_y.size))
scores=np.zeros(num_tests)
c0 = 500 #???
for j in range(num_tests):
    #fit the SVM to the data with different polynomial orders j
    clf = SVC(kernel='poly',max_iter=100000,degree=j,coef0=c0,gamma='auto')
    clf.fit(spiral_x,spiral_y)
    for i in range(spiral_y.size):
        classifications[j][i]=clf.predict([[spiral_x[i][0],spiral_x[i][1]]])
    scores[j]=accuracy_score(spiral_y,classifications[j])
    print("Degree: ", j, " Classification Error Percentage = ", (1-scores[j])*100, "%", "Number of support vectors = ", sum(clf.n_support_))
print("\n\nPolynomial order with maximum accuracy is",np.argmax(scores),"with accuray",scores[np.argmax(scores)], "\n\n")
clf = None
#plot the contour for the best accuracy
clf = SVC(kernel='poly',max_iter=100000,degree=np.argmax(scores),coef0=c0, gamma='auto')
clf.fit(spiral_x,spiral_y)
plot_contours(clf,spiral_x,spiral_y)
plt.title("1c) Contour for the best accuracy ({}) \n using the polynomial kernel with p = {} and a = {}".format(scores[np.argmax(scores)], np.argmax(scores), c0))
print("\n----------------------------------------------------------------------\n")

#1(d) Using the same data try now the Gaussian RBF kernel. Find the approximate
#value of γ for which you can get the best classifier. Comment on the effect of γ
#on your results.
print("\n3.1d)----------------------------------------------------------------------\n")
gammas=np.logspace(start=-4,stop=1,num=num_tests)#create various values of gamma to test
spiral_rbf_classification=np.zeros((gammas.size,spiral_y.size))
score_spiral_rbf_classification = np.zeros(gammas.size)

for j in range(gammas.size):
    clf = SVC(kernel='rbf',max_iter=100000,gamma=gammas[j])
    clf.fit(spiral_x,spiral_y)
    for i in range(spiral_y.size):
            spiral_rbf_classification[j][i]=clf.predict([[spiral_x[i][0],spiral_x[i][1]]])
    score_spiral_rbf_classification[j]=accuracy_score(spiral_y,spiral_rbf_classification[j])
    print("Gamma: ", gammas[j], " Classification Error Percentage = ", (1-score_spiral_rbf_classification[j])*100, "%", "Number of support vectors = ", sum(clf.n_support_))
    clf = None
print("\n\nGaussian RBF gamma with maximum accuracy is",gammas[np.argmax(score_spiral_rbf_classification)],"with accuray",score_spiral_rbf_classification[np.argmax(score_spiral_rbf_classification)], "\n\n")

#plot the contour for the best accuracy
clf = SVC(kernel='rbf',max_iter=100000,gamma=gammas[np.argmax(score_spiral_rbf_classification)])
clf.fit(spiral_x,spiral_y)
plot_contours(clf,spiral_x,spiral_y)
plt.title('Contour for the best accuracy, using the RBF kernel')
plt.title("1d) Contour for the best accuracy ({}) \n using the Gaussian RBF kernel with \n γ = {}".format(score_spiral_rbf_classification[np.argmax(score_spiral_rbf_classification)], gammas[np.argmax(score_spiral_rbf_classification)]))
print("\n----------------------------------------------------------------------\n")
clf = None




#2(a) Load the files chess33_x.npy and chess33_y.npy which contain 90 patterns per
#class arranged in a chess pattern
#use gaussian kernel with chess data
chess_x=np.load('chess33_x.npy')
chess_y=np.load('chess33_y.npy')

"""
#2(b) Use a Gaussian RBF kernel and set ’C’ parameter to Inf to enforce a hard margin
#SVM, for separable data.
chess_autogamma_classf=np.empty(chess_y.size)
#gamma auto
clf = SVC(kernel='rbf',max_iter=100000,gamma='auto',C=math.inf)
clf.fit(chess_x,chess_y)
for i in range(chess_y.size):
        chess_autogamma_classf[i]=clf.predict([[chess_x[i][0],chess_x[i][1]]])
score_chess_rbf_classification=accuracy_score(chess_y,chess_autogamma_classf)
print("Accuracy for the classifier with rbf kernel on the chess data is",score_chess_rbf_classification)
plot_contours(clf,spiral_x,spiral_y)
print("Current Gamma auto is: ",1/(2* chess_x.var()), np.sum(clf.n_support_))
clf = None
"""

#2(c) Find a value of γ that approximately minimizes the number of support vectors,
#while correctly classifying all patterns. Indicate the value of γ and the number of
#support vectors.
print("\n2c)----------------------------------------------------------------------\n")
gammas=np.logspace(start=-4,stop=3,num=num_tests)#create various values of gamma to test
numb_support_vectors=np.empty(gammas.size)#store number of support vector for each gamma here
chess_rbf_classification=np.empty((gammas.size,chess_y.size))
score_chess_rbf_classification=np.empty(gammas.size)
for j in range(gammas.size):
    clf = SVC(kernel='rbf',max_iter=100000,gamma=gammas[j],C=math.inf)#different gammas
    clf.fit(chess_x,chess_y)
    for i in range(chess_y.size):
            chess_rbf_classification[j][i]=clf.predict([[chess_x[i][0],chess_x[i][1]]])
    score_chess_rbf_classification[j]=accuracy_score(chess_y,chess_rbf_classification[j])
    numb_support_vectors[j]=np.sum(clf.n_support_)#number of support vectors_
    print("Gamma: ", gammas[j], " Classification Error Percentage = ", (1-score_chess_rbf_classification[j])*100, "%", "Number of support vectors = ", sum(clf.n_support_))
    clf = None

best_gamma_index=0
best_gamma=gammas[best_gamma_index]
for k in range(gammas.size):
    if(numb_support_vectors[k]<numb_support_vectors[best_gamma_index] and score_chess_rbf_classification[k]==1):
        best_gamma=gammas[k]
        best_gamma_index=k
        
print("\n\nGaussian RBF gamma that minimizes number of support vectors with maximum accuracy is: ",best_gamma,"with accuray",score_chess_rbf_classification[best_gamma_index],'and number of support vectors',numb_support_vectors[best_gamma_index], "\n\n")


clf = SVC(kernel='rbf',max_iter=100000,gamma=best_gamma,C=math.inf)
clf.fit(chess_x,chess_y)
plot_contours(clf,chess_x,chess_y)
plt.title('2c) Contour for the best accuracy ({}) with minimum \n number of support vectors ({}), using the \n Gaussian RBF kernel with γ = {}'.format(score_chess_rbf_classification[best_gamma_index] ,numb_support_vectors[best_gamma_index], best_gamma))
print("\n----------------------------------------------------------------------\n")
clf = None
#3(a)Load the files chess33n_x.npy and chess33n_y.npy which contain data similar
#to the one used in the previous question, except for the presence of a couple of
#outlier patterns.
chess_x_n=np.load('chess33n_x.npy')
chess_y_n=np.load('chess33n_y.npy')

#3(b) Run the classification algorithm on these data with the same value of γ, and
#comment on how the results changed, including the shape of the classification
#border, the margin size and the number of support vectors
clf = SVC(kernel='rbf',max_iter=100000,gamma=best_gamma,C=math.inf)
clf.fit(chess_x_n,chess_y_n)
chess_n_classf=np.empty(chess_y_n.size)
for i in range(chess_y_n.size):
    chess_n_classf[i]=clf.predict([[chess_x_n[i][0],chess_x_n[i][1]]])
score_chess_n_classf=accuracy_score(chess_y_n,chess_n_classf)
plot_contours(clf,chess_x_n,chess_y_n)
plt.title('3b) Contour using the RBF kernel with \n γ = {}, \n accuracy = {} \n and number of support vectors = {}'.format(best_gamma,score_chess_n_classf,sum(clf.n_support_)))
clf = None
print("\n3c)----------------------------------------------------------------------\n")
            
            
#3(c) Now reduce the value of ’C’ parameter in order to obtain the so-called soft margin
#SVM. Try different values (suggestion: use powers of 10). Comment on the results.
C=np.logspace(start=-3,stop=10,num=num_tests)
chess_n_rbf_classification=np.empty((C.size,chess_y_n.size))
score_chess_n_rbf_classification=np.empty(C.size)
score_chess_n_rbf_num_sv=np.empty(C.size)
for j in range(C.size):
    clf = SVC(kernel='rbf',max_iter=100000,gamma=best_gamma,C=C[j])#different gammas
    clf.fit(chess_x_n,chess_y_n)
    for i in range(chess_y_n.size):
            chess_n_rbf_classification[j][i]=clf.predict([[chess_x_n[i][0],chess_x_n[i][1]]])
    score_chess_n_rbf_classification[j]=accuracy_score(chess_y_n,chess_n_rbf_classification[j])
    score_chess_n_rbf_num_sv[j]=sum(clf.n_support_)
    print("C: ", C[j], " Accuracy = ", score_chess_n_rbf_classification[j], "Number of support vectors = ", sum(clf.n_support_))
    clf = None
    
clf = SVC(kernel='rbf',max_iter=100000,gamma=best_gamma,C=C[np.argmax(score_chess_n_rbf_classification)])
clf.fit(chess_x_n,chess_y_n)
plot_contours(clf,chess_x_n,chess_y_n)
plt.title('3c) Contour using the RBF kernel with \n γ = {}, maximum accuracy({}) \n for C = {} \n (and number of support vectors = {})'.format(best_gamma,max(score_chess_n_rbf_classification), C[np.argmax(score_chess_n_rbf_classification)], sum(clf.n_support_)))

best_index = 54
clf = SVC(kernel='rbf',max_iter=100000,gamma=best_gamma,C=C[best_index])
clf.fit(chess_x_n,chess_y_n)
plot_contours(clf,chess_x_n,chess_y_n)
plt.title('3c) Contour using the RBF kernel with \n γ = {}, accuracy = {} \n for C = {} \n (and number of support vectors = {})'.format(best_gamma,score_chess_n_rbf_classification[best_index], C[best_index], sum(clf.n_support_)))



clf = None
