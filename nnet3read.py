#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:12:03 2017
Updated on Tue Sep  5 11:12:03 2017

@author: Omid Sadjadi <s.omid.sadjadi@gmail.com>
"""

""" This is a simple, yet fast, script that reads in Kaldi NNet3 Weight and Bias
    parameters, and converts them into lists of 64-bit floating point numpy arrays
    and optionally dumps the parameters to disk in HDF5 format.
"""

import mmap
import re
import numpy as np
import h5py

write_to_disk =  False

dnn_file_path = "final.txt"
out_file_path = "DNN_1024.h5"
# nn_elements = ['LinearParams', 'BiasParams']
with open(dnn_file_path, 'r') as f:
    pattern = re.compile(rb'<(\bLinearParams\b|\bBiasParams\b)>\s+\[\s+([-?\d\.\de?\s]+)\]')
    with mmap.mmap(f.fileno(), 0,
                   access=mmap.ACCESS_READ) as m:
        b = []
        W = []
        ix = 0
        for arr in pattern.findall(m):
            if arr[0] == b'BiasParams':
                b.append(arr[1].split())
                print("layer{}: [{}x{}]".format(ix, len(b[ix]), len(W[ix])//len(b[ix])))
                ix += 1
            elif arr[0] == b'LinearParams':
                W.append(arr[1].split())
            else:
                raise ValueError('oops... NN element not recognized!')

# converting list of strings into lists of 64-bit floating point numpy arrays and reshaping
b = [np.array(b[ix], dtype=np.float).reshape(-1, 1) for ix in range(len(b))]
W = [np.array(W[ix], dtype=np.float).reshape(len(b[ix]), len(W[ix])//len(b[ix])) for ix in range(len(W))]

if write_to_disk:
    # writing the DNN parameters to an HDF5 file
    with h5py.File(out_file_path, 'w') as h5f:
        for ix in range(len(b)):
            h5f.create_dataset('w'+str(ix), data=np.c_[b[ix], W[ix]], 
            dtype='f8', compression='gzip', compression_opts=9)
