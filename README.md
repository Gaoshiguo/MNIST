# MNIST
#使用MNIST数据集实现手写数字识别
##一、MNIST数据集的介绍以及前期的准备工作

MNIST数据集有很多种格式，常见的有.gz/.npz等等，这里我们选用*.npz*格式的MNIST数据集，.npz数据集中将数据以词典的形式存储。
前期的准备工作需要我们导入**TensorFlow、Keras、matplotlib**等
首先下载MNIST数据集，有两种方法：*直接写代码自动下载或者自己去官网下载数据集然后导入*

```
(x_train_image,y_train_lable),\
(x_test_image,y_test_lable) = mnist.load_data()

```

这行代码就是写代码，通过代码来下载MNIST数据集，但是笔者在实际操作中发现，**由于给的网址是国外的，会报错，无法下载。** ***所以推荐第二种方法：自己去官网上下载MNIST数据集，然后导入***

```
(x_train_image,y_train_lable),\
(x_test_image,y_test_lable) = mnist.load_data('mnist.npz')
```

将数据集中的train、test的image和train、test的lable分别用变量x_train_image、y_train_lable、x_test_image、y_test_lable来表示

```print('train data',len(x_train_image))```#输出有多少张训练集图片

```print('test data',len(x_test_image))```#输出有多少张测试集图片

```print('x_train_image',x_train_image.shape)```#输出训练集的属性

```print('x_test_image',x_test_image.shape)```#输出测试集的属性

![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist-image/1.png)

![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist-image/2.png)

可以看到输出结果显示：***训练集数据60000张、测试集数据10000张***

![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist-image/3.png)

![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist-image/4.png)

**可以看到结果显示为（60000,28,28），代表的意思是六万张训练集图片，每张图片是28x28像素的，同样测试集（10000,28,28）代表的意思是10000张测试集图片
每张28x28**

接下来我们导入matplotlib包，这个包的主要功能是图片处理，可以将训练集和测试集中的图片直观的显示在我们眼前

![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist-image/5.png)

我们定义了一个plot_image（）函数，用于展示图片，括号中image作为参数

`<fig=plt.gcf()>`

`<fig.set_size_inches(2,2)>`

设置图片的大小为2英寸x2英寸

`<plt.imshow(image, cmap='binary')>`/调用plt.image函数显示图形，传入参数image，cmap参数设置为binary，代表以黑白显示

传入的参数为x_train_image[0],显示训练集第一张图片，图片显示如下：

![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist-image/6.png)




