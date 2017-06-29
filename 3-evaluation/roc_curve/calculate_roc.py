# Siamese Architecture for face recognition

import random
import numpy as np
import time
import tensorflow as tf
import math
import pdb
import sys
import scipy.io as sio
from sklearn import *
import matplotlib.pyplot as plt

score = np.load('files/score_vector.npy')
label = np.load('files/target_label_vector.npy')



def calculate_eer_auc_ap(label,distance):

    fpr, tpr, thresholds = metrics.roc_curve(label, distance, pos_label=1)
    AUC = metrics.roc_auc_score(label, distance, average='macro', sample_weight=None)
    AP = metrics.average_precision_score(label, distance, average='macro', sample_weight=None)

    # Calculating EER
    intersect_x = fpr[np.abs(fpr - (1 - tpr)).argmin(0)]
    EER = intersect_x

    return EER,AUC,AP,fpr, tpr

# K-fold validation for ROC
k=1
step = int(label.shape[0] / float(k))
EER_VECTOR = np.zeros((k,1))
AUC_VECTOR = np.zeros((k,1))
for split_num in range(k):
    index_start = split_num * step
    index_end = (split_num + 1) * step
    EER_temp,AUC_temp,AP,fpr, tpr = calculate_eer_auc_ap(label[index_start:index_end],score[index_start:index_end])
    EER_VECTOR[split_num] = EER_temp * 100
    AUC_VECTOR[split_num] = AUC_temp * 100

print("EER=",np.mean(EER_VECTOR),np.std(EER_VECTOR))
print("EER=",np.mean(AUC_VECTOR),np.std(AUC_VECTOR))


