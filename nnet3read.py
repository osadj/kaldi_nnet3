#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:12:03 2017
Updated on Tue Sep  5 13:10:03 2017

@author: Omid Sadjadi <s.omid.sadjadi@gmail.com>
"""

import os
import mmap
import re
import numpy as np
import h5py

def nnet3read(dnnFilename, outFilename="", write_to_disk=False):
    """ This is a simple, yet fast, routine that reads in Kaldi NNet3 Weight and Bias
        parameters, and converts them into lists of 64-bit floating point numpy arrays
        and optionally dumps the parameters to disk in HDF5 format.
        
        :param dnnFilename: input DNN file name (it is assumed to be in text format)
        :param outFilename: output hdf5 filename [optional]
        :param write_to_disk: whether the parameters should be dumped to disk [optional]
        
        :type dnnFilename: string
        :type outFilename: string    
        :type write_to_diks: bool
        
        :return: returns the NN weight and bias parameters (optionally dumps to disk)
        :rtype: tuple (b,W) of list of 64-bit floating point numpy arrays
        
        :Example:
            
        >>> b, W = nnet3read('final.txt', 'DNN_1024.h5', write_to_disk=True)
    """
    # nn_elements = ['LinearParams', 'BiasParams']
    with open(dnnFilename, 'r') as f:
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
        if not outFilename:
            raise ValueError('oops... output file name not specified!')
        filepath = os.path.dirname(outFilename)
        if filepath and not os.path.exists(filepath):
            os.makedirs(filepath)
        with h5py.File(outFilename, 'w') as h5f:
            for ix in range(len(b)):
                h5f.create_dataset('w'+str(ix), data=np.c_[b[ix], W[ix]], 
                dtype='f8', compression='gzip', compression_opts=9)
                
    return b, W

if __name__ == '__main__':
    b, W = nnet3read('final.txt', 'DNN_1024.h5', write_to_disk=True)
