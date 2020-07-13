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


def pariwise(data):
    n = len(data)
    for i in range(n):
        for j in range(i+1, n):
            yield (data[i], data[j])


def krum_aggregate(gradients, f, m=None):

    n = len(gradients)
    if m is None:
        m = n - f - 2

    distances = np.array([0] * (n*(n-1)//2))
    for i, (x, y) in enumerate(pariwise(tuple(range(n)))):
        dist = gradients[x]-gradients[y]
        for i, item in enumerate(dist):
            dist[i] = np.linalg.norm(item)
        dist = np.linalg.norm(dist)
        if not math.isfinite(dist):
            dist = math.inf
        distances[i] = dist

    scores = list()  
    for i in range(n):
        # Collect the distances
        grad_dists = list()
        for j in range(i):
            grad_dists.append(distances[(2 * n - j - 3) * j // 2 + i - 1])
        for j in range(i + 1, n):
            grad_dists.append(distances[(2 * n - i - 3) * i // 2 + j - 1])
        # Select the n - f - 1 smallest distances
        grad_dists.sort()
        scores.append((np.sum(grad_dists[:n - f - 1]), gradients[i], i))
        
    # Compute the average of the selected gradients
    scores.sort(key=lambda x: x[0])
    accepted_nums = [accepted_num for _,_,accepted_num in scores[:m]]

    return [sum(grad for _, grad, _ in scores[:m])/m, accepted_nums]
