import tables
import numpy as np
import os

"""
The function is designed to calculate the mean and std of the training data.
"""
file_name = os.path.join(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')),'development.hdf5')
fileh = tables.open_file(file_name, mode='r')
train_data_shape = fileh.root.utterance_train.shape

size_speech = '80x40x1'
feature = 'mfec'

mean_speech = np.mean(fileh.root.utterance_train, axis=0)
# np.save('mean' + '_' + 'speech' + '_' + feature + '_'+ size_speech + '_' + imp_gep , mean_speech)
np.save('files/'+ 'mean' + '_' + 'speech' + '_' + feature + '_' + size_speech, mean_speech)
print('saving the numpy mean for speech data is over!',mean_speech.shape)


std_speech = np.std(fileh.root.utterance_train, axis=0)
np.save('files/'+ 'std' + '_' + 'speech' + '_' + feature + '_' + size_speech, std_speech)
print('saving the numpy std for speech data is over!',std_speech.shape)



