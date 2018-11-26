# -*- coding: utf-8 -*-
"""
Created on Wed Sep 5 17:54:09 2018

@author: xngu0004
"""

import random
import numpy as np
import scipy.io as sio
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import train_test_split

####################################################
" Reading the data "
print('loading the dataset')

" 0: Holdout, 1: loso "
cv_scheme = 1
" if loso "
user = 5
"Select a configuration"
skill = "mhc3_acc"
#skill = "mhc3_both"
#skill = "mhc3_gyro"

####################################################
# users in Group
novice = ['A', 'B', 'D', 'K', 'L', 'M', 'N']
intermediate = ['E', 'G', 'H']
expert = ['C', 'F', 'I', 'J']

" Load the data"
dataset = sio.loadmat('./data/'+skill+'/dataX4.mat')
features = dataset['dataX']
print('loaded the dataset')

if cv_scheme == 0: # Holdout
    " Initialize data for concatenation "
    X_t = np.ones(((((features[0,0:1])[0])[3,:])[0].shape[0],1))
    y_t = np.ones(1)
    
    # Extract data for training and testing
    for j in range(0,3): # 3 means 3 groups
        for i in range(0, (((features[0,j:j+1])[0])[0,:]).shape[0]):
            temp_x = (((features[0,j:j+1])[0])[3,:])[i]
            X_t = np.concatenate((X_t, temp_x), axis=1) 
            user_temp = (((features[0,j:j+1])[0])[1,:])[i]
        
            if(user_temp in novice):
                y_t = np.concatenate((y_t,[0]))
            elif(user_temp in intermediate):
                y_t = np.concatenate((y_t,[1]))
            elif(user_temp in expert):
                y_t = np.concatenate((y_t,[2]))      
                  
    " Delete the first column "
    X_t = np.delete(X_t,np.s_[0:1], axis=1)
    y_t = np.delete(y_t,np.s_[0:1])
    
    " Shuffle the training and testing data "
    X_train, X_test, y_train, y_test = train_test_split(X_t.transpose(), y_t, test_size=0.2, random_state=random.randint(10,50))
elif cv_scheme == 1: # LOSO
    " Initialize data for concatenation "
    X_test1 = np.ones(((((features[0,0:1])[0])[3,:])[0].shape[0],1))
    y_test1 = np.ones(1)
    X_train1 = np.ones(((((features[0,0:1])[0])[3,:])[0].shape[0],1))
    y_train1 = np.ones(1)

    # Extract data for training and testing
    for j in range(0,3): # 3 means 3 groups
        for i in range(0, (((features[0,j:j+1])[0])[0,:]).shape[0]):
            k = (((features[0,j:j+1])[0])[2,:])[i]
            if (k == user): # take out the ith trial of each user for testing
                temp_x = (((features[0,j:j+1])[0])[3,:])[i]
                X_test1 = np.concatenate((X_test1, temp_x), axis=1) 
                user_temp = (((features[0,j:j+1])[0])[1,:])[i]
        
                if(user_temp in novice):
                    y_test1 = np.concatenate((y_test1,[0]))
                elif(user_temp in intermediate):
                    y_test1 = np.concatenate((y_test1,[1]))
                elif(user_temp in expert):
                    y_test1 = np.concatenate((y_test1,[2]))
            else:
                temp_x = (((features[0,j:j+1])[0])[3,:])[i]
                X_train1 = np.concatenate((X_train1, temp_x), axis=1)
                user_temp = (((features[0,j:j+1])[0])[1,:])[i]
        
                if(user_temp in novice):
                    y_train1 = np.concatenate((y_train1,[0]))
                elif(user_temp in intermediate):
                    y_train1 = np.concatenate((y_train1,[1]))
                elif(user_temp in expert):
                    y_train1 = np.concatenate((y_train1,[2]))
        
    " Delete the first column "
    X_test1 = np.delete(X_test1,np.s_[0:1], axis=1)
    y_test1 = np.delete(y_test1,np.s_[0:1])
    X_train1 = np.delete(X_train1,np.s_[0:1], axis=1)
    y_train1 = np.delete(y_train1,np.s_[0:1])

    " Shuffle the training and testing data "
    X_test, y_test = shuffle(X_test1.transpose(), y_test1, random_state = random.randint(10,50))
    X_train, y_train = shuffle(X_train1.transpose(), y_train1, random_state = random.randint(10,50))

print("Loading done")

#####################################################
" Classification "

cl = svm.SVC(kernel='linear')
cl.fit(X_train, y_train)

predictions = cl.predict(X_test)

print("-------------------------------------------")
print("Skill: ", skill)
if (cv_scheme == 1):
    print("User out:", user)
print('accuracy: ' + str(np.sum(predictions == y_test)/predictions.shape[0]))
print(classification_report(y_test, predictions, digits = 4))
