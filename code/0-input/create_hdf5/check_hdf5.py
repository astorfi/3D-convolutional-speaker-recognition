import tables
import numpy as np
import matplotlib.pyplot as plt

# Reading the file.
fileh = tables.open_file('development.hdf5', mode='r')

# Dimentionality of the data structure.
print(fileh.root.utterance_test.shape)
print(fileh.root.utterance_train.shape)
print(fileh.root.label_train.shape)
print(fileh.root.label_test.shape)


