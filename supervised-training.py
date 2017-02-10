import csv
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import pandas as pd
from ggplot import *
import scipy as sp
import random
import math

my_data = np.genfromtxt('train.csv', delimiter=',')

y = np.array(my_data[:,:1])
y.shape = (y.shape[0],)

from sklearn.preprocessing import label_binarize
y = label_binarize(y, classes=[0, 1,2])
y = np.array(y[:,:2])
n_classes = y.shape[1]

X = np.delete(my_data, 0, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Standardise features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# --------- Logistic Regression ----------- #
from sklearn.linear_model import LogisticRegression
logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_train, y_train)
# predictions
predictions = logreg.predict(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test, predictions)

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')

# logloss
sklearn.metrics.log_loss(y_test,predictions)

# --------------- (scikit) Neural Networks ----------------- #

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(12),max_iter =40)

mlp.fit(X_train,y_train)

# Predictions and model evaluation

predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

sklearn.metrics.log_loss(y_test,predictions)

# ----------------- (keras) Neural Networks ------------------ #

from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential()
model.add(Dense(1776, input_dim=1776, init='uniform', activation='relu'))
model.add(Dense(12, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['msle'])

# Fit the model
model.fit(X_train, y_train, nb_epoch=40, batch_size=10)

# evaluate the model
scores = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X_test)
# round predictions
rounded = [np.round(x) for x in predictions]

predictions = np.int8(np.array(rounded))
predictions.shape = (predictions.shape[0],)
sklearn.metrics.log_loss(Y_test,predictions)

# ------------------- SVMs ------------------------- #

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=True)
clf.fit(X_train, y_train)

# Predictions and model evaluation
predictions = clf.predict_proba(X_test)
predictions = np.array(predictions[:,1])
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
sklearn.metrics.log_loss(y_test,predictions)

# Compute ROC curve and ROC area for each class

fpr, tpr, _ = metrics.roc_curve(y_test, predictions)

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')
