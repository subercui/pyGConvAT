# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:53:44 2018

@author: think
"""

import mne
import os
from mne.io import concatenate_raws
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# load_data
    
def data_augmentation_rolling(x, y):
    num_chs = x.shape[1]
    x_rolled_list = []
    y_rolled_list = []
    x_temp = np.zeros(x.shape)
    y_temp = np.zeros(y.shape)
    for i in range(num_chs):
        x_temp[:,0,:,:] = x[:,-1,:,:]
        x_temp[:,1:,:,:] = x[:,:-1,:,:]
        y_temp[0] = y[-1]
        y_temp[1:] = y[:-1]        
        x = x_temp
        y = y_temp
        x_rolled_list.append(x)
        y_rolled_list.append(y)
    x_rolled = np.concatenate(x_rolled_list, axis = 0)
    y_rolled = np.concatenate(y_rolled_list, axis = 0)
    return x_rolled, y_rolled

from keras.models import Model
from keras.layers import Input, Dense, LSTM, concatenate, Activation, add, BatchNormalization, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten
from keras.layers.merge import multiply, maximum, dot, average
from keras import backend as K
from keras.losses import mean_absolute_percentage_error, hinge
from keras.regularizers import l1, l2
from keras.callbacks import ReduceLROnPlateau
from keras.initializers import glorot_normal
from keras.callbacks import EarlyStopping 
from keras.optimizers import SGD, adam
from keras.layers.advanced_activations import PReLU, ELU

def get_model():
    Input_eeg = Input(shape=(20, 400, 2))
    
    conv1 = Conv2D(5, (3,10), padding = "same", activation = 'relu')(Input_eeg)
    conv1 = Conv2D(5, (3,10), padding = "same", activation = 'relu')(conv1)
    conv1 = MaxPooling2D(pool_size = (2,3))(conv1)
    
    conv2 = Conv2D(5, (3,10), padding = "same", activation = 'relu')(conv1)
    conv2 = Conv2D(5, (3,10), padding = "same", activation = 'relu')(conv2)
    conv2 = MaxPooling2D(pool_size = (2,3))(conv2)
    
    conv3 = Conv2D(5, (3,10), padding = "same", activation = 'relu')(conv2)
    conv3 = Conv2D(5, (3,10), padding = "same", activation = 'relu')(conv3)
    conv3 = MaxPooling2D(pool_size = (2,3))(conv3)
    
    flat_layer = Flatten()(conv3)
    
    dense1 = Dense(10, activation = 'relu')(flat_layer)
    #dense2 = Dense(10, activation = 'relu')(dense1)
    
    output = Dense(1, activation = 'sigmoid')(dense1)
    
    model = Model(inputs = [Input_eeg], outputs = [output])
    
    return model
    
def pred_convert(preds):
    pred_converted = np.zeros(preds.shape)
    for i in range(len(pred_converted)):
        if preds[i] == True:
            pred_converted[i] = 1
    return pred_converted

if __name__ == '__main__':
    
    config = K.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = K.tf.Session(config=config)
    
    d = np.load('dataset1.npz')
    x_pos = d["arr_0"]
    y_pos = d["arr_1"]
    x_neg = d["arr_2"]
    y_neg = d["arr_3"]

    '''
    index_pos = np.arange(x_pos.shape[0])
    np.random.shuffle(index_pos) 
    x_pos = 
    '''
    
    split_rate = 0.7
    x_train = np.concatenate((x_pos[0:int(split_rate * x_pos.shape[0])], x_neg[0:int(split_rate * x_neg.shape[0])]), axis=0)
    x_test = np.concatenate((x_pos[int(split_rate * x_pos.shape[0]):], x_neg[int(split_rate * x_neg.shape[0]):]), axis=0)
    
    y_train = np.concatenate([y_pos[0:int(split_rate * x_pos.shape[0])], y_neg[0:int(split_rate * x_neg.shape[0])]], axis=0)
    y_test = np.concatenate([y_pos[int(split_rate * x_pos.shape[0]):], y_neg[int(split_rate * x_neg.shape[0]):]], axis=0)
    y_whole = np.concatenate([y_train, y_test])
    
    model = get_model()
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    
    NUM_EPOCH = 10
    BATCHSIZE = 64
    
    '''
    x_train = x_train[:,:,:,None]
    x_test = x_test[:,:,:,None]
    '''

    print("data loaded")
    #x_train, y_train = data_augmentation_rolling(x_train, y_train)
    #x_test, y_test = data_augmentation_rolling(x_test, y_test)
    
    TRAIN = True
    
    if TRAIN: 
        history = model.fit(x_train, y_train, \
                        epochs=NUM_EPOCH, batch_size=BATCHSIZE, shuffle=True, validation_data = (x_test, y_test))    
        model.save_weights('weights_conv2d_new.h5') 
        #plt.plot(history.history['loss'])
        #plt.plot(history.history['val_loss'])        
        
    else:
        model.load_weights('weights_conv2d.h5')
    
    #plt.figure(figsize=(18,5))
    preds = model.predict(x_test)
    #plt.plot(preds > 0.5)
    #plt.plot(y_test)
    #plt.axis([0, preds.shape[0], -0.1, 1.1])
    
    from sklearn.metrics import confusion_matrix
    confusion_mat = confusion_matrix(y_test, preds > 0.5)

    print(confusion_mat)
