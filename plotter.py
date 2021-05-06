import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np
import os
from keras.utils import np_utils
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import idx2numpy
import time

def plot_from_inp(inp, m):
	out = model(inp, training = False)
	plt.imshow(out[0, :, :, 0], cmap = 'gray')
	plt.show()


model = tf.keras.models.load_model('mnist_GAN_generator')

while True:
	plot_from_inp(tf.random.normal((1, 100)), model)