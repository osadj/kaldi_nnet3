#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:49:55 2017

@author: Omid Sadjadi <s.omid.sadjadi@gmail.com>
"""

import numpy as np
import nnet3read

def splice_feats(x, w=9):
    """ This routine splices the feature vectors in x by stacking over a window
        of length w frames (must be odd)
    """
    if w < 3 or ((w & 1) != 1):
        raise ValueError('Window length should be an odd integer >= 3')
    hlen = int(w / 2.)
    ndim, nobs = x.shape
    xx = np.c_[np.tile(x[:, 0][:,np.newaxis], hlen), x, np.tile(x[:, -1][:,np.newaxis], hlen)]
    y = np.empty((w*ndim, nobs), dtype=x.dtype)
    for ix in range(w):
        y[ix*ndim:(ix+1)*ndim, :] = xx[:, ix:ix+nobs]
    return y

def renorm_rms(data, target_rms=1.0, axis=0):
    """ This routine scales the data such that the RMS is 1.0
    """
    #scale = 1.0 / sqrt(x^t x / (D * target_rms^2)).
    D = data.shape[axis]
    scale = np.sqrt(np.sum(data * data, axis=axis, keepdims=True)/(D * target_rms * target_rms)) + 0.0
    scale[scale==0] = 1.
    return data / scale

def squashit(aff, nonlin, renorm=False):
    """ This routine applies Sigmoid and RELU activation functions along with the 
        RMS renorm
    """
    if nonlin=='sigmoid':
        aff = sigmoid(aff)
    elif nonlin=='relu':
        np.maximum(aff, 0, aff)
    if renorm:
        aff = renorm_rms(aff, axis=0)
    return aff

def sigmoid(x):
    """ This routine implements Sigmoid nonlinearity
    """
    return 1 / (1 + np.exp(-x))

def extract_bn_features(dnn, fea, nonlin='sigmoid', renorm=False):
    """ This routine computes the bottleneck features using the DNN parameters (b, W)
        and the spliced feature vectors fea. It is assumed that the last layer is 
        the bottleneck layer. This can be achieved by running the following command:
        nnet3-copy --binary=false --nnet-config='echo output-node name=output input=dnn_bn.renorm |' \
                   --edits='remove-orphans' exp/nnet3/swbd9/final.raw exp/nnet3/swbd/final.txt
    """
    b, W = dnn
    aff = fea
    for bi,wi in zip(b[:-1],W[:-1]):
        aff = wi.dot(aff) + bi
        aff = squashit(aff, nonlin, renorm)
    aff = W[-1].dot(aff) + b[-1]
    return aff


if __name__ == '__main__':
    # example that shows how to extract bottleneck features from (say) MFCCs        
    dnn = nnet3read('final.txt', 'DNN_1024.h5', write_to_disk=True)
    
    # we assume mfc is a numpy array of [ndim x nframes] dimesnsion, e.g., [39 x 537]
    # that contains 39-dimensional (say) MFCCs. Features are spliced by stacking over 
    # a 21-frame context
    fea = splice_feats(mfc, w=21)
    
    # now we extract bottleneck features using the DNN parameters and the spliced 
    # features. Here we assume that a RELU ativation function is used, and followed
    # by a renorm nonlinearity to scale the RMS of the vector of activations to 1.0.
    # This kind of nonlinearity is implemented in Kaldi nnet3 as 'relu-renorm-layer'.
    bnf = extract_bn_features(dnn, fea, nonlin='relu', renorm=True)
