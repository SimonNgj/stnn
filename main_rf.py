# -*- coding: utf-8 -*-
"""
Created on Wed Sep 5 17:54:09 2018

@author: xngu0004
"""

import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

####################################################
# Reading the data
print('loading the dataset')

" 0: Holdout, 1: loso "
cv_scheme = 1
" if loso " # from 0 to 4
user = 4
#####################
"Select a skill"
#skill = "acc"
skill = "gyro"
#skill = "acc_gyro"

" 2 or 3 levels" # choose 2 or 3
no_levels = 3;

f1 = open("./data/" + skill + str(no_levels) + "_dataX.csv")
f2 = open("./data/" + skill + str(no_levels) + "_dataY.csv")

data = np.loadtxt(fname = f1, delimiter = ',')
labels = np.loadtxt(fname = f2, delimiter = ',')

if (cv_scheme == 0):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
else:
    X_test = np.zeros((15,data.shape[1]))
    y_test = np.zeros((15,))
    X_train = np.zeros((60,data.shape[1]))
    y_train = np.zeros((60,))
    t_test = 0
    t_train = 0
    for i in range(0,75):
        if ((i%5) == user):
            X_test[t_test] = data[i,:]
            y_test[t_test] = labels[i]
            t_test = t_test + 1
        else:
            X_train[t_train] = data[i,:]
            y_train[t_train] = labels[i]
            t_train = t_train + 1
    X_train, y_train = shuffle(X_train, y_train, random_state = 2)
    X_test, y_test = shuffle(X_test, y_test, random_state = 2)

print("Loading done")

#####################################################
" Classification "

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)

print("-------------------------------------------")
print("Skill: ", skill)
if (cv_scheme == 1):
    print("user out: ", user)
print('accuracy: ' + str(np.sum(predictions == y_test)/predictions.shape[0]))
print(classification_report(y_test, predictions, digits = 4))
