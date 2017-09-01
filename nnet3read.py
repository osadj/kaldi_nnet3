#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:12:03 2017

@author: Omid Sadjadi <s.omid.sadjadi@gmail.com>
"""

""" This is a simple, yet fast, script that reads in Kaldi NNet3 Weight and Bias
    parameters, and converts them into lists of 64-bit floating point numpy arrays
"""

import mmap
import re
import numpy as np

file_path = "final.txt"
# nn_elements = ['LinearParams', 'BiasParams']
with open(file_path, 'r') as f:
    pattern = re.compile(rb'<(\bLinearParams\b|\bBiasParams\b)>\s+\[\s+([-?\d\.\de?\s+]+)\]')
    with mmap.mmap(f.fileno(), 0,
                   access=mmap.ACCESS_READ) as m:
        b = []
        W = []
        ix = 0
        for arr in pattern.findall(m):
            if arr[0].decode() == 'BiasParams':
                b.append(arr[1].decode().split())
                print("layer{}: [{}x{}]".format(ix, len(b[ix]), len(W[ix])//len(b[ix])))
                ix += 1
            elif arr[0].decode() == 'LinearParams':
                W.append(arr[1].decode().split())
            else:
                raise ValueError('oops... NN element not recognized!')

# converting list of strings into lists of 64-bit floating point numpy arrays and reshaping
b = [np.array(b[ix], dtype=np.float).reshape(-1, 1) for ix in range(len(b))]
W = [np.array(W[ix], dtype=np.float).reshape(len(b[ix]), len(W[ix])//len(b[ix])) for ix in range(len(W))]
        

