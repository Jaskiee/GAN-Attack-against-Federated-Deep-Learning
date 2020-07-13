from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# TensorFlow, tf.keras and tensorflow_federated
import tensorflow as tf
from tensorflow import keras
# import tensorflow_federated as tff

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import functools
import glob
import os
import PIL
import time
import math



def median_aggregate(gradients):
    gradient = gradients[0]
    for i in range(len(gradients[0])):
        tmp = np.array(list(np.stack(item[i] for item in gradients)))
        tmp = np.median(tmp,axis=0)
        gradient[i] = tmp

    return gradient
