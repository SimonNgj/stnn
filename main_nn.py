# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:15:00 2018

@author: xngu0004
"""

import numpy as np
import random
import scipy.io as sio
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
from keras.layers import Input, Dense, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, BatchNormalization, Activation, concatenate
from keras.layers import Dropout, LSTM, Permute
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from utils.keras_utils import squeeze_excite_block
import keras.backend as K
from keras.callbacks import Callback

#####################################################
" Reading the data "
print('loading the dataset')

" 0: Holdout, 1: loso "
cv_scheme = 1
" if loso "
user = 1
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
dataset = sio.loadmat('./data/'+skill+'/dataX.mat')

features = dataset['dataX']
sizeD_ts = (((features[0,0:1])[0])[0,:])[0].shape[0]
sizeD_va = (((features[0,0:1])[0])[0,:])[0].shape[1]
print('loaded the dataset')

if cv_scheme == 0: # Holdout
    " Initialize data for concatenation "
    X_t = np.ones((1,sizeD_ts,sizeD_va))
    y_t = np.ones(1)
    
    # Extract data for training and testing
    for j in range(0,3): # 3 means 3 groups
        for i in range(0, (((features[0,j:j+1])[0])[0,:]).shape[0]):
            temp_x = (((features[0,j:j+1])[0])[0,:])[i]
            temp_x1 = temp_x.reshape(1,sizeD_ts,sizeD_va)
            X_t = np.concatenate((X_t, temp_x1), axis=0) 
            user_temp = (((features[0,j:j+1])[0])[1,:])[i]
        
            if(user_temp in novice):
                y_t = np.concatenate((y_t,[0]))
            elif(user_temp in intermediate):
                y_t = np.concatenate((y_t,[1]))
            elif(user_temp in expert):
                y_t = np.concatenate((y_t,[2]))      
                
    " Delete the first column "
    X_t = np.delete(X_t,np.s_[0:1], axis=0)
    y_t = np.delete(y_t,np.s_[0:1])
    
    " Shuffle the training and testing data "
    X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, random_state=random.randint(10,50))
elif cv_scheme == 1: # LOSO
    " Initialize data for concatenation "
    X_test1 = np.ones((1,sizeD_ts,sizeD_va))
    y_test1 = np.ones(1)
    X_train1 = np.ones((1,sizeD_ts,sizeD_va))
    y_train1 = np.ones(1)

    # Extract data for training and testing
    for j in range(0,3): # 3 means 3 groups
        for i in range(0, (((features[0,j:j+1])[0])[0,:]).shape[0]):
            k = (((features[0,j:j+1])[0])[2,:])[i]
            if (k == user): # take out the ith trial of each user for testing
                temp_x = (((features[0,j:j+1])[0])[0,:])[i]
                temp_x1 = temp_x.reshape(1,sizeD_ts,sizeD_va)
                X_test1 = np.concatenate((X_test1, temp_x1), axis=0) 
                user_temp = (((features[0,j:j+1])[0])[1,:])[i]
        
                if(user_temp in novice):
                    y_test1 = np.concatenate((y_test1,[0]))
                elif(user_temp in intermediate):
                    y_test1 = np.concatenate((y_test1,[1]))
                elif(user_temp in expert):
                    y_test1 = np.concatenate((y_test1,[2]))
            else:
                temp_x = (((features[0,j:j+1])[0])[0,:])[i]
                temp_x1 = temp_x.reshape(1,sizeD_ts,sizeD_va)
                X_train1 = np.concatenate((X_train1, temp_x1), axis=0)
                user_temp = (((features[0,j:j+1])[0])[1,:])[i]
        
                if(user_temp in novice):
                    y_train1 = np.concatenate((y_train1,[0]))
                elif(user_temp in intermediate):
                    y_train1 = np.concatenate((y_train1,[1]))
                elif(user_temp in expert):
                    y_train1 = np.concatenate((y_train1,[2]))
        
    " Delete the first column "
    X_test1 = np.delete(X_test1,np.s_[0:1], axis=0)
    y_test1 = np.delete(y_test1,np.s_[0:1])
    X_train1 = np.delete(X_train1,np.s_[0:1], axis=0)
    y_train1 = np.delete(y_train1,np.s_[0:1])

    " Shuffle the training and testing data "
    X_test, y_test = shuffle(X_test1, y_test1, random_state = random.randint(10,50))
    X_train, y_train = shuffle(X_train1, y_train1, random_state = random.randint(10,50))
    
    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]
    
" One-hot coding"
labels_train = to_categorical(y_train)
labels_test = to_categorical(y_test)

class LRrestart(Callback):
    def __init__(self, min_lr, max_lr, steps_per_epoch, lr_decay=1, cycle_length=10, mult_factor=2):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay
        self.batch_since_restart = 0
        self.next_restart = cycle_length
        self.steps_per_epoch = steps_per_epoch
        self.cycle_length = cycle_length
        self.mult_factor = mult_factor
        self.history = {}

    def clr(self):
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weights)
        
####################################################
print("Ready to train")
" creating CNN "
def model_arch():
    ip = Input(shape=(sizeD_ts,sizeD_va), name='main_input')
    #x1 = Permute((2, 1))(ip)
    x1 = LSTM(8, return_sequences=True)(x1)
    x1 = LSTM(8)(x1)

    x2 = Conv1D(16, 7, padding='same', kernel_initializer='he_uniform')(ip)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = squeeze_excite_block(x2)

    x2 = Conv1D(32, 5, padding='same', kernel_initializer='he_uniform')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = GlobalAveragePooling1D()(x2)

    x = concatenate([x1, x2])
    out = Dense(3, kernel_regularizer=regularizers.l2(0.001), activation='softmax')(x)

    model = Model(ip, out)
    model.summary()
    return model

if __name__ == "__main__":
    
    model = model_arch()
    epochs_s = 100
    batch_s = 32
    learning_rate = 1e-3   
    weight_fn = "./weights_mhc/weights.h5"
    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode='max', monitor='val_acc', save_best_only=True, save_weights_only=True)
    lr_restart = LRrestart(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=np.ceil(epochs_s/batch_s), lr_decay=0.9, cycle_length=5, mult_factor=1.5)
#    callback_list = [model_checkpoint, lr_restart]
    callback_list = [model_checkpoint]
    optm = Adam(lr=learning_rate)

    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, labels_train, batch_size=batch_s, epochs=epochs_s, callbacks = callback_list, class_weight=class_weight, verbose=2, validation_data=(X_test, labels_test))

    model.load_weights(weight_fn)
    prediction = model.predict(X_test);
    y_pred = np.argmax(prediction, axis=1)
    
    print("-------------------------------------------")
    print("Skill: ", skill)
    if (cv_scheme == 1):
        print("User out:", user)
    cr = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    acc = np.sum(y_pred == y_test)/y_pred.shape[0]
    print('Accuracy:' + str(acc))
    print(cr)
    print(cm)
    f = open(skill+'_'+str(user)+'_report.txt', 'w')
    f.write('-------------' + skill + ": " + str(user) + '----------------\n\n')
    f.write('Accuracy:' + str(acc))
    f.write('\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
    f.close()
