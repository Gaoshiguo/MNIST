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
def plot_images_lables_prediction(images,lables,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx], cmap = 'binary')
        title = "lable="+str(lables[idx])
        if len(prediction)>0:
            title+=",prediction="+str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
plot_images_lables_prediction(x_test_image,y_test_lable,[],0,10)

x_Train =x_train_image.reshape(60000,784).astype('float32')
x_Test =x_test_image.reshape(10000,784).astype('float32')
print('x_train',x_Train)
print('x_test',x_Test)

x_Train_normalize =x_Train/255
x_Test_normalize =x_Test/255
print('x_train_normalize',x_Train_normalize)
print('x_test_normalize',x_Test_normalize)


print('train data',len(x_train_image))
print('test data',len(x_test_image))
print('x_train_image',x_train_image.shape)
print('x_test_image',x_test_image.shape)