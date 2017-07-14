"""
This function is designed to create HDF5 file for the development phase.
The process for creating the data files for enrollment and evaluation phases are similar.

DEVELOPMENT: The development set itself is divided to two sets of train and test.
             The test set is just used for online testing of the accuracy in the development set and will have no more effects!

             utterance_train: Its size is [num_training_samples,num_frames,num_coefficient,num_utterances]
                              num_frames: The number of windows frames the the features are extracted from them. default=80
                              num_coeficients: Number of energy coefficients (features) for each window. default=40
                              num_utterances: number of utterances for each speaker in development and enrollment phases. default=20

SAMPLE_DATA: The provided sample data is not the dataset itself. They are the extracted features.
             Each .npy file in the sample dataset is a feature_vector from the sound file.
             The feature vector has the dimentionality of [num_frames,num_coefficients,num_channels]

             num_frames: The whole frames for the sound file.
             num_coefficients: Number of energy coefficients (features) for each window. default=40
             num_channels: The static, first and second order derivatives of speech features. default=3

Feature_extraction:

                 The feature extraction stage is prior to this phase.
                 The speech features must be extracted from each sound file and be stored as numpy files
                 as with the aformentioned dimensionality.

                 The following package has been used for speech feature extraction.

                 @misc{amirsina_torfi_2017_810392,
                      author       = {Amirsina Torfi},
                      title        = {astorfi/speech_feature_extraction: SpeechPy},
                      month        = jun,
                      year         = 2017,
                      doi          = {10.5281/zenodo.810392},
                      url          = {https://doi.org/10.5281/zenodo.810392}
                      }



"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import random
import tables
import pickle
from pair_generation import feed_to_hdf5
from random import shuffle


#########################################
########### TRAIN/TEST HDF5 #############
#########################################
phase = 'development'

output_filename = phase + '.hdf5'
#################################################
################# HDF5 elements #################
#################################################

# DEFAULTS:
num_frames = 80
num_coefficient = 40
num_utterances = 20

hdf5_file = tables.open_file(output_filename, mode='w')
filters = tables.Filters(complevel=5, complib='blosc')
utterance_train_storage = hdf5_file.create_earray(hdf5_file.root, 'utterance_train',
                                         tables.Float32Atom(shape=(), dflt=0.0),
                                         shape=(0, num_frames, num_coefficient, num_utterances),
                                         filters=filters)
utterance_test_storage = hdf5_file.create_earray(hdf5_file.root, 'utterance_test',
                                         tables.Float32Atom(shape=(), dflt=0.0),
                                         shape=(0, num_frames, num_coefficient, num_utterances),
                                         filters=filters)
label_train_storage = hdf5_file.create_earray(hdf5_file.root, 'label_train',
                                        tables.IntAtom(shape=(), dflt=0.0),
                                        shape=(0,),
                                        filters=filters)
label_test_storage = hdf5_file.create_earray(hdf5_file.root, 'label_test',
                                        tables.IntAtom(shape=(), dflt=0.0),
                                        shape=(0,),
                                        filters=filters)


########################################
###### parsing text file ###############
########################################
dataset_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'sample_data'))
IDs_list = 'train_subjects_path.txt'

# Get the subjects dirs.
with open(IDs_list) as f:
    content = f.readlines()

# Get the subject ids.
IDs = [os.path.basename(x.strip()) for x in content]


##############################################
############ Training subjects ###############
##############################################

## Information ###
TotalNumSubjects = len(IDs)


# Keep the first 100 subjects for enrollment phase.
train_subjects = IDs[0:TotalNumSubjects]


# Get the training files!
train_files_list = []
train_files_subjects_list = []
train_files_subjects_ids = []


# For each subject get all the files, read and store the file name in another list.
# This is crucial for shuffling!
for subject_num, subject in enumerate(train_subjects):
    path_subject = os.path.join(dataset_path,subject)
    for root, dirs, files in os.walk(path_subject, topdown=False):
        for name in files:
            file_name = os.path.join(root, name)
            train_files_list.append(file_name)
            train_files_subjects_list.append(file_name.split('/')[7])
            train_files_subjects_ids.append(subject_num)


##############################################
######### Shuffle the training files##########
##############################################
NumTrainFiles = len(train_files_list)

list_file = []
list_subject = []
list_id = []
index_shuf = range(len(train_files_subjects_list))
shuffle(index_shuf)


for i in index_shuf:
    list_file.append(train_files_list[i])
    list_subject.append(train_files_subjects_list[i])
    list_id.append(train_files_subjects_ids[i])


##############################################
######### Feeding to HDF5 structure ##########
##############################################
# Get all the files and fed them to HDF5 structure.
for counter, (file_name, subject, id) in enumerate(zip(list_file,list_subject,list_id)):
    feature_vector = np.load(file_name)
    feed_to_hdf5(feature_vector, id, utterance_train_storage, utterance_test_storage, label_train_storage, label_test_storage)
    if (counter+1) % 10 == 0:
        print('Processing %d-th file of %d' % (counter+1,NumTrainFiles))
