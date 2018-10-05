# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 17:54:09 2018

@author: xngu0004
"""

import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

####################################################
# Reading the data
print('loading the dataset')

" 0: Holdout, 1: loso "
cv_scheme = 0 
" if loso "
user = 1
"Select a skill"
#skill = "acc"
#skill = "gyro"
skill = "acc_gyro"
" 2 or 3 levels"
no_levels = 3;

if (cv_scheme == 0):
    f1 = open("./data/" + skill + str(no_levels) + "_dataX.csv")
    f2 = open("./data/" + skill + str(no_levels) + "_dataY.csv")

    data = np.loadtxt(fname = f1, delimiter = ',')
    labels = np.loadtxt(fname = f2, delimiter = ',')

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
else:
    f1_train = open("./data/"+skill+"_dataX_" + str(user) + "_train.csv")
    f2_train = open("./data/"+skill+"_dataY_" + str(user) + "_train.csv")
    f1_test = open("./data/"+skill+"_dataX_" + str(user) + "_test.csv")
    f2_test = open("./data/"+skill+"_dataY_" + str(user) + "_test.csv")

    X_train = np.loadtxt(fname = f1_train, delimiter = ',')
    y_train = np.loadtxt(fname = f2_train, delimiter = ',')
    X_test = np.loadtxt(fname = f1_test, delimiter = ',')
    y_test = np.loadtxt(fname = f2_test, delimiter = ',')

print("Loading done")

####################################################
" Classification "

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)

print("-------------------------------------------")
print("Skill: ", skill)
print('accuracy: ' + str(np.sum(predictions == y_test)/predictions.shape[0]))
print(classification_report(y_test, predictions, digits = 4))