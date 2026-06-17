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
window_size = 9
numComponents = 15

data = applyPCA(data, numComponents = numComponents, normalization = True)

dummy = np.zeros((1, window_size, window_size, numComponents, 1), dtype=np.float32)
model = SS_MixNet(dummy, mixer_dim=128, num_classes=NUM_CLASS)


model.load_weights(f"{DATASET}_SS_MixNet.h5")

Predicted_Class_Map = get_class_map(model, data, gt, window_size)
img_display(classes=Predicted_Class_Map, title='Predicted', class_name=class_name)

gt_binary = gt.copy()
gt_binary[gt>0]=1
img_display(classes=Predicted_Class_Map*gt_binary, title='Predicted with Mask', class_name=class_name)


Folder = 'Matlab_Outputs/'
Name = f'SS_MixNet'
sio.savemat(Folder + DATASET+'/' + Name+'.mat', {Name: Predicted_Class_Map})