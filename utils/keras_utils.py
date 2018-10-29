import os
import numpy as np
import scipy.io as sio
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils.data_set import TRAIN_FILES, EVAL_FILES, MAX_NB_VARIABLES
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import GlobalAveragePooling1D, Reshape, Dense, multiply

def train_model(model:Model, dataset_id, dataset_fold_id=None, epochs=50, batch_size=32, val_subset=None, normalize_timeseries=False, learning_rate=1e-3, monitor='val_acc', optimization_mode='max', compile_model=True):
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id, fold_index=dataset_fold_id, normalize_timeseries=normalize_timeseries)

    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    print("Class weights : ", class_weight)

    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    factor = 1. / np.sqrt(2)

    if dataset_fold_id is None:
        weight_fn = "./weights/%s_weights.h5" % dataset_id
    else:
        weight_fn = "./weights/%s_fold_%d_weights.h5" % (dataset_id, dataset_fold_id)

    model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=optimization_mode, monitor=monitor, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=100, mode=optimization_mode, factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    callback_list = [model_checkpoint, reduce_lr]

    optm = Adam(lr=learning_rate)

    if compile_model:
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if val_subset is not None:
        X_test = X_test[:val_subset]
        y_test = y_test[:val_subset]

    lr_restart = LRrestart(lr_0=0.01,  time_steps=2, cycle_length=5)
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
              class_weight=class_weight, verbose=2, validation_data=(X_test, y_test))

def eval_model(model:Model, dataset_id, dataset_fold_id=None, batch_size=32, normalize_timeseries=False):
    _, _, X_test, y_test, is_timeseries = load_dataset_at(dataset_id, fold_index=dataset_fold_id, normalize_timeseries=normalize_timeseries)

    if not is_timeseries:
        X_test = pad_sequences(X_test, maxlen=MAX_NB_VARIABLES[dataset_id], padding='post', truncating='post')
    y_test = to_categorical(y_test, len(np.unique(y_test)))

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    if dataset_fold_id is None:
        weight_fn = "./weights/%s_weights.h5" % dataset_id
    else:
        weight_fn = "./weights/%s_fold_%d_weights.h5" % (dataset_id, dataset_fold_id)
    model.load_weights(weight_fn)

    prediction = model.predict(X_test);
    y_pred = np.argmax(prediction, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print('Accuracy: ' + str(np.sum(y_pred == y_true)/y_pred.shape[0]))
    print(classification_report(y_true, y_pred))   

def squeeze_excite_block(input):
    filters = input._keras_shape[-1] 
    seb = GlobalAveragePooling1D()(input)
    seb = Reshape((1, filters))(seb)
    seb = Dense(filters // 8,  activation='relu', kernel_initializer='he_normal', use_bias=False)(seb)
    seb = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(seb)
    seb = multiply([input, seb])
    return seb

def load_dataset_jigsaws(cv_scheme, user, skill):
    dataset = sio.loadmat('./data/'+skill+'_dataX.mat')
    features = dataset['dataX']
    novice = ['B', 'G', 'H', 'I']
    intermediate = ['C', 'F']
    expert = ['D', 'E']
    if cv_scheme == 0: # Holdout
        
        X_t = np.ones((1,480,76))
        y_t = np.ones(1)
    
        # Extract data for training and testing
        for j in range(0,3): # 3 means 3 groups
            for i in range(0, (((features[0,j:j+1])[0])[0,:]).shape[0]):
                temp_x = (((features[0,j:j+1])[0])[0,:])[i]
                temp_x1 = temp_x.reshape(1,480,76)
                X_t = np.concatenate((X_t, temp_x1), axis=0) 
                user_temp = (((features[0,j:j+1])[0])[1,:])[i]
        
                if(user_temp in novice):
                    y_t = np.concatenate((y_t,[0]))
                elif(user_temp in intermediate):
                    y_t = np.concatenate((y_t,[1]))
                elif(user_temp in expert):
                    y_t = np.concatenate((y_t,[2]))      
                
        X_t = np.delete(X_t,np.s_[0:1], axis=0)
        y_t = np.delete(y_t,np.s_[0:1])
    
        X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, random_state=10)
    elif cv_scheme == 1: # LOSO

        X_test1 = np.ones((1,480,76))
        y_test1 = np.ones(1)
        X_train1 = np.ones((1,480,76))
        y_train1 = np.ones(1)

        # Extract data for training and testing
        for j in range(0,3): # 3 means 3 groups
            for i in range(0, (((features[0,j:j+1])[0])[0,:]).shape[0]):
                k = (((features[0,j:j+1])[0])[2,:])[i]
                if (k == user): # take out the ith trial of each user for testing
                    temp_x = (((features[0,j:j+1])[0])[0,:])[i]
                    temp_x1 = temp_x.reshape(1,480,76)
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
                    temp_x1 = temp_x.reshape(1,480,76)
                    X_train1 = np.concatenate((X_train1, temp_x1), axis=0)
                    user_temp = (((features[0,j:j+1])[0])[1,:])[i]
        
                    if(user_temp in novice):
                        y_train1 = np.concatenate((y_train1,[0]))
                    elif(user_temp in intermediate):
                        y_train1 = np.concatenate((y_train1,[1]))
                    elif(user_temp in expert):
                        y_train1 = np.concatenate((y_train1,[2]))
        
        X_test1 = np.delete(X_test1,np.s_[0:1], axis=0)
        y_test1 = np.delete(y_test1,np.s_[0:1])
        X_train1 = np.delete(X_train1,np.s_[0:1], axis=0)
        y_train1 = np.delete(y_train1,np.s_[0:1])

        X_test, y_test = shuffle(X_test1.transpose(), y_test1, random_state = 10)
        X_train, y_train = shuffle(X_train1.transpose(), y_train1, random_state = 10)
    
def load_dataset_at(index, fold_index=None, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    if verbose: print("Loading train / test dataset : ", TRAIN_FILES[index], EVAL_FILES[index])

    if fold_index is None:
        x_train_path = TRAIN_FILES[index] + "X_train.npy"
        y_train_path = TRAIN_FILES[index] + "y_train.npy"
        x_test_path = EVAL_FILES[index] + "X_test.npy"
        y_test_path = EVAL_FILES[index] + "y_test.npy"
    else:
        x_train_path = TRAIN_FILES[index] + "X_train_%d.npy" % fold_index
        y_train_path = TRAIN_FILES[index] + "y_train_%d.npy" % fold_index
        x_test_path = EVAL_FILES[index] + "X_test_%d.npy" % fold_index
        y_test_path = EVAL_FILES[index] + "y_test_%d.npy" % fold_index

    if os.path.exists(x_train_path):
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
    elif os.path.exists(x_train_path[1:]):
        X_train = np.load(x_train_path[1:])
        y_train = np.load(y_train_path[1:])
        X_test = np.load(x_test_path[1:])
        y_test = np.load(y_test_path[1:])
    else:
        raise FileNotFoundError('File %s not found!' % (TRAIN_FILES[index]))

    is_timeseries = True

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_train_mean = X_train.mean()
            X_train_std = X_train.std()
            X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

    print("Finished traning")

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)

    print("Finished loading test data..")
    return X_train, y_train, X_test, y_test, is_timeseries

def load_dataset_at_simon(index, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    if verbose: print("Loading new dataset : ", EVAL_FILES[index])

    x_test_path = EVAL_FILES[index] + "X_new_test.npy"
    y_test_path = EVAL_FILES[index] + "y_new_test.npy"

    if os.path.exists(x_test_path):
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
    elif os.path.exists(x_test_path[1:]):
        X_test = np.load(x_test_path[1:])
        y_test = np.load(y_test_path[1:])
    else:
        raise FileNotFoundError('File %s not found!' % (EVAL_FILES[index]))

    is_timeseries = True

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_test_mean = X_test.mean()
            X_test_std = X_test.std()
            X_test = (X_test - X_test_mean) / (X_test_std + 1e-8)

    print("Finished traning")
    return X_test, y_test

def load_dataset_at_Xsimon(index, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    if verbose: print("Loading new dataset : ", EVAL_FILES[index])

    x_test_path = EVAL_FILES[index] + "X_new_test.npy"

    if os.path.exists(x_test_path):
        X_your_test = np.load(x_test_path)
    elif os.path.exists(x_test_path[1:]):
        X_your_test = np.load(x_test_path[1:])
    else:
        raise FileNotFoundError('File %s not found!' % (EVAL_FILES[index]))

    is_timeseries = True

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_test_mean = X_your_test.mean()
            X_test_std = X_your_test.std()
            X_your_test = (X_your_test - X_test_mean) / (X_test_std + 1e-8)

    print("Finished traning")
    return X_your_test

class LRrestart(Callback):
    def __init__(self, lr_0=0.01, time_steps=2, cycle_length=5):
        
        self.lr_0 = lr_0       
        self.epoch_since_restart = 0
        self.next_restart = cycle_length
        
        self.time_steps = time_steps    
        self.cycle_length = cycle_length
        self.history = {}
        
    def clr(self):
        if self.epoch_since_restart % self.time_steps == 0:
            k = (self.epoch_since_restart//self.time_steps) % self.cycle_length
            fraction_to_restart = 1 / (2**(k+1) + 5*k)
            lr = self.lr_0 * np.sin(fraction_to_restart * np.pi)
            return lr

    def on_train_begin(self, logs={}):
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.lr_0)

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        if epoch + 1 == self.next_restart:
            self.epoch_since_restart = 0
            self.next_restart += self.cycle_length
            self.best_weights = self.model.get_weights()
        else:
            self.epoch_since_restart += 1
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weights)