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
import os

def Plot_HIST_Fn(label,distance, save_path, num_bins = 50):

    dissimilarity = distance[:]
    gen_dissimilarity_original = []
    imp_dissimilarity_original = []
    for i in range(len(label)):
        if label[i] == 1:
            gen_dissimilarity_original.append(dissimilarity[i])
        else:
            imp_dissimilarity_original.append(dissimilarity[i])

    bins = np.linspace(np.amin(distance), np.amax(distance), num_bins)
    fig = plt.figure()
    plt.hist(gen_dissimilarity_original, bins, alpha=0.5, facecolor='blue', normed=False, label='gen_dist_original')
    plt.hist(imp_dissimilarity_original, bins, alpha=0.5, facecolor='red', normed=False, label='imp_dist_original')
    plt.legend(loc='upper right')
    plt.title('OriginalFeatures_Histogram.jpg')
    plt.show()
    fig.savefig(save_path)

if __name__ == '__main__':
   
    tf.app.flags.DEFINE_string(
    'evaluation_dir', '../../results/SCORES',
    'Directory where checkpoints and event logs are written to.')
    
    tf.app.flags.DEFINE_string(
    'plot_dir', '../../results/PLOTS',
    'Directory where plots are saved to.')
    
    tf.app.flags.DEFINE_integer(
    'num_bins', '50',
    'Number of bins for plotting histogram.')

    # Store all elemnts in FLAG structure!
    FLAGS = tf.app.flags.FLAGS
    
    # Loading necessary data.
    score = np.load(os.path.join(FLAGS.evaluation_dir,'score_vector.npy'))
    label = np.load(os.path.join(FLAGS.evaluation_dir,'target_label_vector.npy'))
    save_path = os.path.join(FLAGS.plot_dir,'Histogram.jpg')
    
    # Creating the path
    if not os.path.exists(FLAGS.plot_dir):
            os.makedirs(FLAGS.plot_dir)
            
    Plot_HIST_Fn(label,score, save_path, FLAGS.num_bins)

