import os

import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.patches as mpts
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from scipy.io import loadmat
from tqdm import tqdm
#from model_MixerFormer import MixerFormer
from tensorflow.keras.callbacks import ModelCheckpoint
#from display_history import display_history
import scipy.io as sio

DATASET = 'Pingan'  ## Pingan, Tangdaowan, Qingyun

data, gt, class_name = loadData(DATASET)
NUM_CLASS = gt.max()
img_display(classes=gt, title='groundtruth', class_name=class_name)


data = applyPCA(data, numComponents = 15, normalization = True)

# Get class map indexes
indexes, labels = get_img_indexes(gt, removeZeroindexes = True)



X_train_idx, X_test_idx, y_train, y_test = splitTrainTestSet(indexes, labels, testRatio = 0.98)
X_train_idx, X_val_idx, y_train, y_val = splitTrainTestSet(X_train_idx, y_train, testRatio = 0.50)

sample_report = f"{'class': ^25}{'train_num':^10}{'val_num': ^10}{'test_num': ^10}{'total': ^10}\n"
for i in np.unique(gt):
    if i == 0: continue
    sample_report += f"{class_name[i-1]: ^25}{(y_train==i-1).sum(): ^10}{(y_val==i-1).sum(): ^10}{(y_test==i-1).sum(): ^10}{(gt==i).sum(): ^10}\n"
sample_report += f"{'total': ^25}{len(y_train-1): ^10}{len(y_val): ^10}{len(y_test): ^10}{len(labels): ^10}"
print(sample_report)


window_size = 9
X_train = createImageCubes(data, X_train_idx, window_size)
y_train = keras.utils.to_categorical(y_train)

X_val = createImageCubes(data, X_val_idx, window_size)
y_val = keras.utils.to_categorical(y_val)

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv3D, Dense, Dropout, Reshape, Permute,
    Add, DepthwiseConv2D,
    SeparableConv2D, Multiply, Activation, GlobalAveragePooling2D,
    GlobalAveragePooling3D, Concatenate
)
from tensorflow.keras.models import Model

def spectral_mixer_block(x, hidden_dim, num_blocks = 4):
    H, W, C, D = x.shape[1:]

    x = Permute((1,2,4,3))(x)
    x = Reshape((H*W, D, C))(x)     
    
    for _ in range(num_blocks):
        x1 = Dense(hidden_dim)(x)     
        x1 = Dense(C)(x1)              
        x = Add()([x, x1])             
    
    out = Reshape((H, W, C, D))(x)   
    
    return out

def spatial_mixer_block(x, hidden_dim, num_blocks = 4):
    H, W, C, D = x.shape[1:]  
    
    x = Permute((3, 4, 1, 2))(x)         
    x = Reshape((C, D, H * W))(x)           

    for _ in range(num_blocks):
        x1 = Dense(hidden_dim)(x)          
        x1 = Dense(H*W)(x1)                  
        x = Add()([x, x1])                 
    
    x = Permute((3, 1, 2))(x)           
    out = Reshape((H, W, C, D))(x)        
    
    return out 


def dw_attention_block(x):
    shape = tf.shape(x)
    B, H, W, C, F = shape[0], shape[1], shape[2], shape[3], shape[4]

    x_2D = tf.reshape(x, (B, H, W, C * F))

    # Apply DepthwiseConv2D + sigmoid gating
    x_dw = DepthwiseConv2D(kernel_size=3, padding='same')(x_2D)
    x_dw = Activation('sigmoid')(x_dw)

    x_2D = Multiply()([x_2D, x_dw])
    
    return x_2D



def SS_MixNet(img_list, mixer_dim=128, num_classes=NUM_CLASS):
    input_shape = img_list.shape[1:] + (1,)
    inp = Input(shape=input_shape)  

    # 3D Convolution Layers
    x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(inp)
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(x)
    
    # Mixer blocks
    # Spectral Mixer Block
    x_spe = spectral_mixer_block(x, mixer_dim)
    
    # Spatial Mixer Block
    x_spa = spatial_mixer_block(x, mixer_dim)
    
    # Feature Fusion
    x = Concatenate(axis= 4)([x_spe, x_spa])
    
    # Attention
    x = dw_attention_block(x)
    
    
    
    # Classification Head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = SS_MixNet(X_train , mixer_dim=128, num_classes=NUM_CLASS)
model.summary()


checkpoint = ModelCheckpoint(
    f"{DATASET}_SS_MixNet.h5",
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Define a callback to modify the learning rate dynamically
lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=10,
    min_lr=5e-5
    )
    
history = model.fit(X_train, y_train,
                    epochs = 100,
                    batch_size = 64,
                    validation_data = (X_val, y_val),
                    callbacks=[checkpoint, lr_callback],
                    )

        
Y_pred_test = predict_by_batching(model, input_tensor_idx = X_test_idx, batch_size = 1000, X = data, windowSize = window_size)
y_pred_test = np.argmax(Y_pred_test, axis=1)
    
kappa = cohen_kappa_score(y_test,  y_pred_test)
oa = accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test)
class_acc = cm.diagonal() / cm.sum(axis=1)
aa = np.mean(class_acc)
    
 
print("Overall Accuracy = ", float(format((oa)*100, ".2f"))) 
print("Average Accuracy = ", float(format((aa)*100, ".2f")))
print('Kappa = ', float(format((kappa)*100, ".2f")))
 


model.load_weights(f"{DATASET}_SS_MixNet_{i}.h5")

Predicted_Class_Map = get_class_map(model, data, gt, window_size)
img_display(classes=Predicted_Class_Map, title='Predicted', class_name=class_name)

gt_binary = gt.copy()
gt_binary[gt>0]=1
img_display(classes=Predicted_Class_Map*gt_binary, title='Predicted with Mask', class_name=class_name)


Folder = 'Matlab_Outputs/'
Name = f'SS_MixNet'
sio.savemat(Folder + DATASET+'/' + Name+'.mat', {Name: Predicted_Class_Map})















