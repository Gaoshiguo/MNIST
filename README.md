# MNIST
# 使用MNIST数据集实现手写数字识别

## 一、MNIST数据集的介绍以及前期的准备工作

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
## 二、MNIST数据集的读取和查看
将数据集中的train、test的image和train、test的lable分别用变量`<x_train_image>`、`<y_train_lable>`、`<x_test_image>`、`<y_test_lable来表示>`

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
我们又定义了一个`<plot_images_lables_prediction()>`函数用来展示更多的图片，函数参入的参数有**images(数字图像)、label(真实值)、prediction(预测结果)、idx(展示的第一张图片序号)、num(想展示的图片数，默认是10，不可以超过25张) 
** 
该函数的代码片段为

``` 
def plot_images_lables_prediction(images,lables,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)#设置图片显示的大小
    if num>25: num=25#设置显示图片的数目小于25 ，如果超过25就设置为25
    for i in range(0,num):#for循环执行程序内的代码块，画出num个数字图形
        ax = plt.subplot(5,5,1+i)#建立子图为5行5列
        ax.imshow(images[idx], cmap = 'binary')#画出子图，黑白色显示
        title = "lable="+str(lables[idx])#显示子图的label
        if len(prediction)>0:#如果传入了预测结果
            title+=",prediction="+str(prediction[idx])#就显示预测结果
        ax.set_title(title,fontsize=10)#设置子图的标题
        ax.set_xticks([]);ax.set_yticks([])#设置不显示刻度
        idx+=1#读取下一项
    plt.show()#画出子图
plot_images_lables_prediction(x_test_image,y_test_lable,[],0,10)#为函数传入参数，分别画出测试集的图片和显示测试集的label，这里没有传入预测值，所以传入空列表，从第0张开始画出十张
```  

完整代码如下图：  
![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist-image/7.png)  
运行结果如下图：  
![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist-image/8.png)

# 三、数据预处理（提取特征值features）
1.将原本28x28的图像数据调用reshape()方法，转换为以为的向量，由于是28x28，所以转换的一维向量长度是784，类型为浮点类型Float
`<x_Train =x_train_image.reshape(60000,784).astype('float32')
x_Test =x_test_image.reshape(10000,784).astype('float32')
print('x_train',x_Train)
print('x_test',x_Test)>`  
![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist/9.png)
![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist/10.png)  
运行代码后可以发现：`<x_Test>`和`<x_Train>`分别存储了训练集和测试集数据的一维向量信息

2.对label数据进行预处理  
label数据是一串0~9的数字，我们需要将其也转换为一个个的一维向量，例如0可以转换为
[1,0,0,0,0,0,0,0,0,0],代表的意思是第0和数字为1，那么该向量表示的数字为0，1可以转换为[0,1,0,0,0,0,0,0,0,0,0]，代表的意思是第一个数字为1，该向量代表的数字为1，同理2可以表示为[0,0,1,0,0,0,0,0,0,0,0],我们可以调用`<np_utils.to_categorcally()>`方法来实现，具体代码为：  
```
y_TrainOnehot = np_utils.to_categorical(y_train_lable)
y_TestOnehot = np_utils.to_categorical(y_test_lable)
print(y_TrainOnehot[:5])
print(y_TestOnehot[:5])

```
实际代码及运行结果图如下所示：  
![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist/12.png)  
该图显示了label中前五个个数据分别为5,0,4,1,9  
![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist/13.png)  
![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist/14.png)  
该图反映了在经过转换以后的各个label变成了一维向量来存储信息
# 四、使用Keras多层感知器模型来进行训练
*多层感知器模型的介绍*
多层感知器模型包括输入层、隐藏层、输出层。

**结合本次实例，输入层就是将二维图像转化成的一维向量，是一组28x28=784的一维向量，隐藏层是256个神经元，输出层是10个输出神经元，因为输出层是0-9这10个数字，我们是将这十个数字转化成一个10个位点的一维向量，所以输出层是一个十个神经元的一维向量** 

**4.1建立线性模型**   

`<model = Sequential()>`该行代码可以建立一个线性堆叠模型，后续只需使用`model.add()`方法就可以添加各个层  

**4.2加入“输入层”和“隐藏层” **   
```
model.add(Dense(units=256,
                input_dim=784,
                kernel_initializer='normal',
                activation="relu"))
```
这部分代码就是将*输入层*和*隐藏层*添加至模型中，其中各个参数的含义如下：**添加隐藏层模型，nuits代表隐藏层神经元有256个，input_dim代表输入层有784个参数，kernel_initializer代表使用正态分布随机初始化权值，activation定义激活函数为relu** 

**4.3加入"输出层" ** 
```
model.add(Dense(units=10,
           kernel_initializer="normal",
           activation="softmax"))
```
 这部分代码是将“输出层”添加进model模型中，各参数的含义如下：**建立输出层模型，units代表输出层神经元10个，kernel_initializer代表使用正态分布随机初始化权值，activation代表定义激活函数为softmax**
 
 使用`print(model.summary())`来查看模型的摘要
 
 运行结果如下图：  
![image](https://github.com/Gaoshiguo/MNIST/blob/master/mnist-image/9.png)

我们分析代码运行结果，可以看到有两个层，分别是dense_1和dense_2两个层,输入层和隐藏层是一起建立的，所以没有显示输入层，在dense_1后有一个256，代表隐藏层有256个神经元，dense_1有10，代表输出层有10个神经元，后面还有param,代表参数个数，由于是线性堆叠模型，所以参数是线性关系，通过反向传播算法更新神经元连接的权重与偏差，输入层到隐藏层的公式为：*h1=relu(X x W1 +b1)*,隐藏层到输出层的公式为：*y=softmax(h1 x W2 +b2)*,由此可以得到每一层的param的计算方式就是（上一层神经元个数）x（本层神经元个数）+（本层神经元数量）.可以看到隐藏层的Param=200960，其计算方法是784(输入层神经元个数)x256（隐藏层神经元个数）+256（隐藏层神经元个数）。输出层Param=2570,2570=256（隐藏层神经元个数）x10(输出层神经元个数)+10(输出层神经元个数)

**4.4进行训练**
4.4.1定义训练方式：在训练模型之前必须使用compile方法对训练模型进行设置
```
model.compile(
                loss="categorical_crossentropy",
                optimizer='adam',
                metrics=['accuracy'])
 ```
#各参数的含义如下：
#loss:设置损失函数，使用categorical_crossentropy（交叉熵）训练效果比较好
#optimizer：使用adam优化器，让训练更快收敛，提高准确率
#metrics：设置评估模型的方式是准确率

4.4.2开始训练
```
train_history =model.fit(x=x_Train_normalize,
                         y=y_TrainOnehot,validation_split=0.2,
                         epochs=10,batch_size=200,verbose=2)
```

使用model.fit方法进行训练，将训练的结果存储在train_history变量中，各个参数的含义如下：
`x=x_Train_normalize`feature数字图像的特征值
`y=y_TrainOnehot`label数字图像的真实值`validation_split=0.2`设置训练与验证数据的比例，将80%用作训练数据，20%用作验证数据，`epochs=10`设置训练周期，一共10个训练周期，`batch_size=200`每个周期训练200项数据，`verbose=2`显示训练过程

代码运行结果如下：








