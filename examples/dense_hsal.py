#!/usr/bin/python

"""
Adapted from the original C++ example: densecrf/examples/dense_inference.cpp
http://www.philkr.net/home/densecrf Version 2.2
"""

import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from skimage.segmentation import relabel_sequential
import sys

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if len(sys.argv) != 4:
    print("Usage: python {} IMAGE ANNO OUTPUT".format(sys.argv[0]))
    print("")
    print("IMAGE and ANNO are inputs and OUTPUT is where the result should be written.")
    sys.exit(1)

EPSILON = 1e-8

img = cv2.imread(sys.argv[1], 1)
annos = cv2.imread(sys.argv[2], 0)
labels = relabel_sequential(cv2.imread(sys.argv[2], 0))[0].flatten()
output = sys.argv[3]

M = 2  # salient or not
tau = 1.05
# Setup the CRF model
d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

anno_norm = annos / 255.
n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))

U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
U[0, :] = n_energy.flatten()
U[1, :] = p_energy.flatten()

d.setUnaryEnergy(U)

d.addPairwiseGaussian(sxy=3, compat=3)
d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

# Do the inference
infer = np.array(d.inference(1)).astype('float32')
res = infer[1,:]

#res *= 255 / res.max()
res = res * 255
res = res.reshape(img.shape[:2])
cv2.imwrite(output, res.astype('uint8'))
