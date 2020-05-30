import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils import print_summary 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, balanced_accuracy_score

'''
Functions used to plot the Metrics and the ROC curve
'''

def plot_roc(labels, predictions): 
    fp, tp, _ = roc_curve(labels, predictions) 
    plt.figure() 
    plt.title('ROC') 
    plt.plot(fp, tp, label='ROC') 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance') 
    plt.grid(True) 
    ax = plt.gca() 
    ax.set_aspect('equal') 
    plt.legend(loc="lower right") 
    plt.show()

def plot_metrics(history):
  metrics =  ['loss', 'auc', 'precision', 'recall']
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  for n, metric in enumerate(metrics):
    plt.figure()
    name = metric.replace("_"," ").capitalize()
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])
      
    plt.legend()
    plt.show()
    
'''
DataSet1-> .Balanced Dataset
'''

print("DATASET 1")

dataset1_xtrain=np.load('dataset1_xtrain.npy')
dataset1_ytrain=np.load('dataset1_ytrain.npy')
dataset1_xtest=np.load('dataset1_xtest.npy')
dataset1_ytest=np.load('dataset1_ytest.npy')

def build_MLP_model(train_data, metrics): 
    model = keras.Sequential([ keras.layers.Dense(units=64, activation='relu', input_shape=(train_data.shape[-1],)), keras.layers.Dense(units=128, activation='sigmoid'), keras.layers.Dense(units=1, activation='sigmoid') ]) 
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=keras.losses.BinaryCrossentropy(), metrics=metrics) 
    return model

METRICS = [ keras.metrics.TruePositives(name='tp'), keras.metrics.FalsePositives(name='fp'), keras.metrics.TrueNegatives(name='tn'), keras.metrics.FalseNegatives(name='fn'), keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')] 
# Start training the model, validate the model with validation data (splitted)
x_train, x_val, y_train, y_val = train_test_split(dataset1_xtrain, dataset1_ytrain, test_size = 0.3) 
model = build_MLP_model(x_train, METRICS)
print_summary(model)

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True) 
# patience = 15
fit = model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=400, batch_size=300, verbose=0, callbacks=[es]) 
print("- Graphics for dataset 1")
plot_metrics(fit)
# Making predictions #
print("\n------------- EVALUATING MODEL WITH TEST DATA -------------\n")
y_pred = (model.predict(dataset1_xtest) > 0.5) * 1 
print(" - Accuracy score: ", accuracy_score(dataset1_ytest, y_pred))
print(" - Confusion matrix: \n", confusion_matrix(dataset1_ytest, y_pred))
print(" - Balanced Accuracy: ",balanced_accuracy_score(dataset1_ytest,y_pred))
plot_roc(dataset1_ytest, y_pred)
print("\n")

'''
DataSet2 -> .Unbalanced Dataset
            .Features also need to be scaled for better learning
'''

print("DATASET 2")
dataset2_xtrain=np.load('dataset2_xtrain.npy')
dataset2_ytrain=np.load('dataset2_ytrain.npy')
dataset2_xtest=np.load('dataset2_xtest.npy')
dataset2_ytest=np.load('dataset2_ytest.npy')

METRICS = [ keras.metrics.TruePositives(name='tp'), keras.metrics.FalsePositives(name='fp'), keras.metrics.TrueNegatives(name='tn'), keras.metrics.FalseNegatives(name='fn'), keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')] 
EPOCHS=400
BATCH_SIZE=300

def make_weighted_model(x_train,metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias=keras.initializers.Constant(output_bias)
    model=keras.Sequential([keras.layers.Dense(128,activation='relu',
                            input_shape=(x_train.shape[-1],)),
                            keras.layers.Dense(128,activation='sigmoid'),
                            keras.layers.Dense(1,activation='sigmoid',bias_initializer=output_bias),
                            ])
    
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3)
    ,loss=keras.losses.BinaryCrossentropy()
    ,metrics=metrics)
    return model

#scale the features
scaler=StandardScaler()
dataset2_xtrain=scaler.fit_transform(dataset2_xtrain)
dataset2_xtest=scaler.fit_transform(dataset2_xtest)

#calculate the frequency of appearance of each class, based on the train dataset
aux=dataset2_ytrain.ravel()
neg,pos=np.bincount(aux.astype(int))
total=neg+pos

#Set the correct Weightes for the classes
weight_for_0=(1/neg)*total/2.0
weight_for_1=(1/pos)*total/2.0
class_weight={0:weight_for_0, 1:weight_for_1}

#Set the correct initial biases for the perceptrons 
initial_bias=np.log([pos/neg])

early_stopping=tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

x_train, x_val, y_train, y_val = train_test_split(dataset2_xtrain, dataset2_ytrain, test_size = 0.3) 

weighted_model=make_weighted_model(x_train,output_bias=initial_bias)
print_summary(weighted_model)

weighted_model.load_weights('weights', by_name=False)

weighted_history=weighted_model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=EPOCHS, 
                batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stopping]
                ,class_weight=class_weight)

print("- Graphics for dataset 2")
plot_metrics(weighted_history)

scores=weighted_model.predict(dataset2_xtest)
y_pred = (scores > 0.5)*1

print("\n------------- EVALUATING MODEL WITH TEST DATA -------------\n")
print(" - Accuracy score: ", accuracy_score(dataset2_ytest, y_pred))
print(" - Confusion matrix: \n", confusion_matrix(dataset2_ytest, y_pred))
print(" - Balanced Accuracy: ",balanced_accuracy_score(dataset2_ytest,y_pred))
plot_roc(dataset2_ytest, y_pred)
print("\n")

'''
To make the various training runs more comparable, keep the initial 
model's weights in a checkpoint file, 
and load them into each model before training.
'''
weighted_model.save_weights('weights') #saves the weights of the model as a HDF5 file.

