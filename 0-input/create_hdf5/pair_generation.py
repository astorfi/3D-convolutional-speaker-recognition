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


def feed_to_hdf5(feature_vector, subject_num, utterance_train_storage, utterance_test_storage, label_train_storage,
                 label_test_storage):
    """

    :param feature_vector: The feature vector for each sound file of shape: (num_frames,num_features_per_frame,num_channles.)
    :param subject_num: The subject class in 'int' format.
    :param utterance_storage: The HDF5 object for storing utterance feature map.
    :param label_train_storage: The HDF5 object for storing train label.
    :param label_test_storage: The HDF5 object for storing test label.
    :return: Each utterance will be stored in HDF5 file.
    """
    num_utterances_per_speaker = 20
    stride_step = 20
    utterance_length = 80
    num_frames = feature_vector.shape[0]
    num_samples = int(np.floor((num_frames - utterance_length - num_utterances_per_speaker) / float(stride_step))) + 1

    # Half of the samples will be fed for training.
    range_training = range(int(4 * num_samples / 5))
    range_training = range(1)

    for sample_index in range_training:
        # initial index of each utterance
        init = sample_index * stride_step
        utterance = np.zeros((1, 80, 40, 20), dtype=np.float32)
        for utterance_speaker in range(num_utterances_per_speaker):
            utterance[:, :, :, utterance_speaker] = feature_vector[None,
                                                    init + utterance_speaker:init + utterance_speaker + utterance_length,
                                                    :, 0]
        utterance_train_storage.append(utterance)
        label_train_storage.append((np.array([subject_num + 1], dtype=np.int32)))

    # The second half of each sound file will be used for testing on the same subject.
    range_testing = range(int(4 * num_samples / 5), int(num_samples))
    range_testing = range(1,2)
    for sample_index in range_testing:
        # initial index of each utterance
        init = sample_index * stride_step
        utterance = np.zeros((1, 80, 40, 20), dtype=np.float32)
        for utterance_speaker in range(num_utterances_per_speaker):
            utterance[:, :, :, utterance_speaker] = feature_vector[None,
                                                    init + utterance_speaker:init + utterance_speaker + utterance_length,
                                                    :, 0]
        utterance_test_storage.append(utterance)
        label_test_storage.append((np.array([subject_num + 1], dtype=np.int32)))
