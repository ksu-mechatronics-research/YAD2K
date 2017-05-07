'''calculate k means for boxes in our underwater dataset'''
import os

import PIL.Image
import h5py

import numpy as np

DATA_PATH = os.path.expanduser(os.path.join('..', 'DATA', 'underwater.hdf5'))
data = h5py.File(DATA_PATH, 'r')

image = PIL.Image.fromarray(data['train/images'][0])
orig_size = np.array([image.width, image.height])
orig_size = np.expand_dims(orig_size, axis=0)
print(data['val/boxes'][13].shape)
boxes = list(data['train/boxes']) + list(data['val/boxes'])

boxes = [box[:][1:] for box in boxes] # remove labels
print(len(boxes[1][0]))
boxes_extents = [box[:][1, 0, 3, 2] for box in boxes] #rearrange
print(len(boxes))
