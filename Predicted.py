import os

import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.patches as mpts
from utils import *
from Main_SS_MixNet import SS_MixNet
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

model = SS_MixNet(X_train , mixer_dim=128, num_classes=NUM_CLASS)
model.summary()


model.load_weights(f"{DATASET}_SS_MixNet.h5")

Predicted_Class_Map = get_class_map(model, data, gt, window_size)
img_display(classes=Predicted_Class_Map, title='Predicted', class_name=class_name)

gt_binary = gt.copy()
gt_binary[gt>0]=1
img_display(classes=Predicted_Class_Map*gt_binary, title='Predicted with Mask', class_name=class_name)


Folder = 'Matlab_Outputs/'
Name = f'SS_MixNet'
sio.savemat(Folder + DATASET+'/' + Name+'.mat', {Name: Predicted_Class_Map})