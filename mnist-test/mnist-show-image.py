import tensorflow as tf
import numpy as np
import keras
import os
import tensorflow.examples.tutorials.mnist.input_data as input_data
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
(x_train_image,y_train_lable),\
(x_test_image,y_test_lable) = mnist.load_data('mnist.npz')
#定义函数来展示图片
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap='binary')
    plt.show()
#展示训练集中的第一张图片
plot_image(x_train_image[0])
print('train data',len(x_train_image))
print('test data',len(x_test_image))
print('x_train_image',x_train_image.shape)
print('x_test_image',x_test_image.shape)