from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, LSTM, concatenate, Activation, Masking, Dropout
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, Permute
from utils.data_set import MAX_NB_VARIABLES, NB_CLASSES, MAX_TIMESTEPS
from utils.keras_utils import train_model,eval_model
from utils.keras_utils import squeeze_excite_block

DATASET_INDEX = 4

MAX_TIMESTEPS = MAX_TIMESTEPS[DATASET_INDEX]
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES[DATASET_INDEX]

def model_arch():
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    x1 = Masking()(ip)
    x1 = LSTM(12, return_sequences=True)(x1)
    x1 = Dropout(0.2)(x1)
    x1 = LSTM(12)(x1)

    x2 = Permute((2, 1))(ip)
    x2 = Conv1D(64, 9, padding='same', kernel_initializer='he_uniform')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.2)(x2)
    x2 = squeeze_excite_block(x2)

    x2 = Conv1D(32, 5, padding='same', kernel_initializer='he_uniform')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.2)(x2)
    x2 = GlobalMaxPooling1D()(x2)

    x = concatenate([x1, x2])
    out = Dense(NB_CLASS, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), activation='softmax')(x)

    model = Model(ip, out)
    model.summary()
    return model

if __name__ == "__main__":
    model = model_arch()
    
    train_model(model, DATASET_INDEX, epochs=50, batch_size=32)
    eval_model(model, DATASET_INDEX, batch_size=32)
    
