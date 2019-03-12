import tensorflow as tf
import numpy as np
import keras
import os
import tensorflow.examples.tutorials.mnist.input_data as input_data
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
(x_train_image,y_train_lable),\
(x_test_image,y_test_lable) = mnist.load_data('mnist.npz')
#定义函数来展示图片
#def plot_image(image):
    #fig = plt.gcf()
    #fig.set_size_inches(2,2)
    #plt.imshow(image,cmap='binary')
    #plt.show()
#展示训练集中的第一张图片
#plot_image(x_train_image[0])


#print(y_train_lable[:5])#展示训练集label标签的前五项数据


y_TrainOnehot = np_utils.to_categorical(y_train_lable)
y_TestOnehot = np_utils.to_categorical(y_test_lable)
x_Train = x_train_image.reshape(60000,784).astype('float32')
x_Test =x_test_image.reshape(10000,784).astype('float32')
x_Train_normalize =x_Train/255
x_Test_normalize =x_Test/255
#建立多层感知器模型
model = Sequential()#先建立一个线性堆叠模型，后面只需要将各个神经网络层使用add方法添加即可
model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation="relu"))
#添加隐藏层模型，nuits代表隐藏层神经元有256个，input_dim代表输入层有784个参数，kernel_initializer
#代表使用正态分布随机初始化权值，activation定义激活函数为relu
model.add(Dense(units=10,kernel_initializer="normal",activation="softmax"))
#建立输出层模型，units代表输出层神经元10个，kernel_initializer代表使用正态分布随机初始化权值
#activation代表定义激活函数为softmax
#使用反向传播算法进行训练模型

#定义训练方式：
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

#开始训练
train_history =model.fit(x=x_Train_normalize,
                         y=y_TrainOnehot,validation_split=0.2,
                         epochs=10,batch_size=200,verbose=2)






print(model.summary())#查看模型的摘要

print(y_TrainOnehot[:5])
print(y_TestOnehot[:5])



print('train data',len(x_train_image))
print('test data',len(x_test_image))
print('x_train_image',x_train_image.shape)
print('x_test_image',x_test_image.shape)