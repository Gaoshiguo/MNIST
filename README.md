# MNIST
使用MNIST数据集实现手写数字识别
一、MNIST数据集的介绍以及前期的准备工作

MNIST数据集有很多种格式，常见的有.gz/.npz等等，这里我们选用.npz格式的MNIST数据集，.npz数据集中将数据以词典的形式存储。
前期的准备工作需要我们导入TensorFlow、Keras、matplotlib等
首先下载MNIST数据集，有两种方法：直接写代码自动下载或者自己去官网下载数据集然后导入

(x_train_image,y_train_lable),\
(x_test_image,y_test_lable) = mnist.load_data()

这行代码就是写代码，通过代码来下载MNIST数据集，但是笔者在实际操作中发现，由于给的网址是国外的，会报错，无法下载。所以推荐第二种方法：自己去官网上下载MNIST数据集，然后导入

(x_train_image,y_train_lable),\
(x_test_image,y_test_lable) = mnist.load_data('mnist.npz')

将数据集中的train、test的image和train、test的lable分别用变量x_train_image、y_train_lable、x_test_image、y_test_lable来表示

print('train data',len(x_train_image))#输出有多少张训练集图片

print('test data',len(x_test_image))#输出有多少张测试集图片

print('x_train_image',x_train_image.shape)#输出训练集的属性

print('x_test_image',x_test_image.shape)#输出测试集的属性

