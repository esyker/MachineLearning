# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:56:44 2019

@author: Diogo
"""

import pandas
import numpy as np
import sklearn.naive_bayes as nb
import sklearn.metrics as skmetrics
import sklearn.feature_extraction as fe

#3.2.1.1. Load the data containing the trigram counts for each language. 
#Note that for tsv files the separator is tab.
eng_trigrams = pandas.read_csv("en_trigram_count.tsv", sep = '\t', header = None)
pt_trigrams = pandas.read_csv("pt_trigram_count.tsv", sep = '\t', header = None)
es_trigrams = pandas.read_csv("es_trigram_count.tsv", sep = '\t', header = None)
fr_trigrams = pandas.read_csv("fr_trigram_count.tsv", sep = '\t', header = None)

#3.2.1.2. Check data format and contents
print("\nData format: ", eng_trigrams.shape)
print("\nData content: \nS", eng_trigrams.head)

#3.2.1.3. Build a training matrix X_train where each language corresponds to one sample and
#with as many features as the number of trigrams. Create the corresponding vector
#y_train containing a different label for each language.
X_train = np.vstack((eng_trigrams[2], pt_trigrams[2], es_trigrams[2], fr_trigrams[2]))#counts of each feature (trigram) in each class (language)

y_train = np.array(['en','pt','es','fr']) #labels for each class

#3.2.1.4. Instantiate a Multinomial Naive Bayes model. Use Laplace smoothing and assume
#equal priors for all languages.
clf = nb.MultinomialNB(alpha = 1, fit_prior = False) #alpha = 1 -> Laplace smoothing, fit_prior = False -> uniform priors

#3.2.1.5. Fit the model using the X_train and y_train matrices you created.
clf.fit(X_train, y_train)

#3.2.2.1. To verify that the recognizer is operating properly make predictions for your training
#data and calculate the accuracy of class predictions.
y_train_predict = clf.predict(X_train)
print("\nAccuracy for training set: ", skmetrics.accuracy_score(y_train, y_train_predict))

#3.2.2.2. Build a sentences matrix containing the test sentences from the table.
test_sentences = ['El cine esta abierto.', 'Tu vais à escola hoje.', 'Tu vais à escola hoje pois já estás melhor.','English is easy to learn.','Tu vas au cinéma demain matin.','É fácil de entender.']

#3.2.2.3. Instantiate a vectorizer in order to obtain the trigram counts for each sentence in
#your sentences matrix. Use as vocabulary the set of trigrams from the training data.
trigrams = eng_trigrams[1]
v = fe.text.CountVectorizer(ngram_range = (3,3), vocabulary=trigrams, analyzer='char')

#3.2.2.4. Learn the data trigram counts for the given sentences and store them in X_test matrix.
#Build the corresponding y_test vector.
X_test = v.fit_transform(test_sentences).toarray()
y_test = np.array(['es','pt','pt','en','fr','pt'])

#3.2.2.5. Make predictions for your test data using the Naive Bayes model.
y_predict_test = clf.predict(X_test)

#3.2.2.6. Compute the classification margin, which is the difference between the two highest
#probabilities.
y_test_predict_prob = clf.predict_proba(X_test)
y_test_predict_prob.sort()
scores = y_test_predict_prob[:,y_test_predict_prob.shape[1]-1] #maximum values
classification_margins = y_test_predict_prob[:,y_test_predict_prob.shape[1]-1] - y_test_predict_prob[:,y_test_predict_prob.shape[1]-2]

print("Real Language \t|\t Recognized Language \t|\t\t Score \t\t|\t Classification Margin \t\t|\t\t Text")
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
for i in range(y_test.size):
    print(y_test[i]," \t\t|\t ", y_predict_test[i]," \t\t\t|\t ", scores[i]," \t|\t ", classification_margins[i], " \t\t|\t ", test_sentences[i] )