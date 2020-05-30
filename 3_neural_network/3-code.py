# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:48:27 2019

@author: Diogo
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
import sklearn.model_selection as ms
import sklearn.metrics as skmetrics

#1.1.1 Load the data files and check the size of inputs X and labels y
X_train = np.load('mnist_train_data.npy')
print("\nX train:", X_train.shape)
y_train = np.load('mnist_train_labels.npy')
print("y train:", y_train.shape)

X_test = np.load('mnist_test_data.npy')
print("X test:", X_test.shape)
y_test = np.load('mnist_test_labels.npy')
print("y train:", y_test.shape)

#1.1.2 Display some of the digits in the train and test data
plt.figure()
plt.imshow(np.squeeze(X_train, axis=3)[1])
plt.figure()
plt.imshow(np.squeeze(X_test, axis=3)[1])

#1.1.3 Divide data by 255 to get floating point values in the 0–1 range
X_train = X_train/255
X_test = X_test/255

#1.1.4 Convert labels to one-hot encoding. In this representation, the label matrix will have
#10 elements, with the component that corresponds to the pattern’s class equal to 1,
#and all the other components equal to 0
y_test = keras.utils.to_categorical(y_test, num_classes=10)
y_train = keras.utils.to_categorical(y_train, num_classes=10)

#1.1.5 Split train data into two subsets, one for actual training and the other for validation.
#Use 30% for validation.
X_train, X_validation, y_train, y_validation = ms.train_test_split(X_train, y_train, test_size=0.3)


#1.2.1 Create a sequential model and start by adding a flatten layer to convert the 2D images
#to 1D. Since this is the first layer in your model, you’ll need to to specify the input
#shape which, for 28x28 pixels with only one color channel (grayscale), is (28,28,1).
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28,1)))

#1.2.2 Add two hidden layers to your MLP, the first with 64 and the second with 128 neurons.
#Use ’relu’ as activation function for all neurons in the hidden layers.
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))

#1.2.3 End the network with a softmax layer. This is a dense layer that will return an array
#of 10 probability scores, one per class, summing to 1
model.add(keras.layers.Dense(10, activation='softmax'))

#1.2.4 Get the summary of your network to check it is correct.
print("\n\nMPL Summary:")
print(model.summary())

#1.2.5 Create an Early stopping monitor that will stop training when the validation loss is
#not improving (use patience=15 and restore_best_weights=True).
earlyStop = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)

#1.2.6 Fit the MLP to your training and validation data using ’categorical_crossentropy’ as
#the loss function, a batch size of 300 and Adam as the optimizer (learning rate=0.01,
#clipnorm=1). Choose, as stopping criterion, the number of iterations reaching 400.
#Don’t forget the Early Stopping callback.
adams = keras.optimizers.Adam(lr=0.01, clipnorm=1)
model.compile(optimizer=adams, loss='categorical_crossentropy')
history = model.fit(x=X_train, y=y_train, validation_data=(X_validation, y_validation), batch_size=300, callbacks = [earlyStop], epochs=400, verbose = 0)


#1.2.7 Plot the evolution of the training loss and the validation loss. Note that the call to
#fit() returns a History object where metrics monitored during training are kept. 
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (MPL with Early Stop)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#1.2.8 To get an idea of how well the model generalizes to new, unseen data, 
#evaluate performance (accuracy and confusion matrix) on the test data.
y_predict = model.predict(X_test, batch_size=300)
print("\n\nAccuracy (MPL with Early Stop):", skmetrics.accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)))
print("Confusion Matrix (MPL with Early Stop):\n", skmetrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_predict, axis=1)))

#1.2.9 Repeat the previous items without Early Stopping.
model2 = keras.Sequential()
model2.add(keras.layers.Flatten(input_shape=(28,28,1)))

model2.add(keras.layers.Dense(64, activation='relu'))
model2.add(keras.layers.Dense(128, activation='relu'))

model2.add(keras.layers.Dense(10, activation='softmax'))

#print(model2.summary())

adams = keras.optimizers.Adam(lr=0.01, clipnorm=1)
model2.compile(optimizer=adams, loss='categorical_crossentropy')
history2 = model2.fit(x=X_train, y=y_train, validation_data=(X_validation, y_validation), batch_size=300, epochs=400, verbose = 0)

# summarize history for loss
plt.figure()
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model Loss (MPL without Early Stop)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

y_predict2 = model2.predict(X_test, batch_size=300)
print("\n\nAccuracy (MPL without Early Stop):", skmetrics.accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predict2, axis=1)))
print("Confusion Matrix (MPL without Early Stop):\n", skmetrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_predict2, axis=1)))


#1.3.1 Create a Convolutional Neural Network (CNN) with two alternated Convolutional
#(with relu activation and 3x3 filters) and MaxPooling2D layers (2x2). Use 16 filters in
#the first conv layer and 32 in the second. Add a flatten layer and then a dense layer with
#64 units (with relu activation). End the network with a softmax layer.
modelCNN = keras.Sequential()
modelCNN.add(keras.layers.Conv2D(16,(3,3), activation='relu', input_shape=(28,28,1)))
modelCNN.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

modelCNN.add(keras.layers.Conv2D(32,(3,3), activation='relu'))
modelCNN.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

modelCNN.add(keras.layers.Flatten())
modelCNN.add(keras.layers.Dense(64, activation='relu'))

modelCNN.add(keras.layers.Dense(10, activation='softmax'))

#1.3.2 Get the summary of your network to check it is correct
print("\n\nCNN Summary:")
print(modelCNN.summary())

#1.3.3 Fit the CNN to your training and validation data. Use the same loss function, batch
#size, optimizer and Early Stopping callback that were used for the MLP.
earlyStop = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)

adams = keras.optimizers.Adam(lr=0.01, clipnorm=1)
modelCNN.compile(optimizer=adams, loss='categorical_crossentropy')
historyCNN = modelCNN.fit(x=X_train, y=y_train, validation_data=(X_validation, y_validation), batch_size=300, callbacks = [earlyStop], epochs=400, verbose = 0)

#1.3.4 Plot the evolution of the training loss and the validation loss
# summarize history for loss
plt.figure()
plt.plot(historyCNN.history['loss'])
plt.plot(historyCNN.history['val_loss'])
plt.title('Model Loss (CNN with Early Stop)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#1.3.5 Evaluate performance (accuracy and confusion matrix) on the test data.
y_predictCNN = modelCNN.predict(X_test, batch_size=300)
print("\n\nAccuracy (CNN with Early Stop):", skmetrics.accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_predictCNN, axis=1)))
print("Confusion Matrix (CNN with Early Stop):\n", skmetrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_predictCNN, axis=1)))
