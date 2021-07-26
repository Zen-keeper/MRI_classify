# MRI_classify

#### 介绍
用于对MRI图像进行处理分类，采用方案为通过切片并拼接，使用2D分类器进行分类，适当修改可以使用Resnet3D分类


#### 使用说明
config.py 配置

train.py 运行

## 图像分类集成以下模型：ResNet18、ResNet34、ResNet50、ResNet101、ResNet152、VGG16、VGG19、InceptionV3、Xception、MobileNet、AlexNet、LeNet、ZF_Net、DenseNet、mnist_net、TSL16，在config.py里面选择使用哪种模型.

## the project apply the following models:


* VGG16
* VGG19
* InceptionV3
* Xception
* MobileNet
* AlexNet
* LeNet
* ZF_Net
* ResNet18
* ResNet34
* ResNet50
* ResNet101
* ResNet152
* DenseNet(dismissed this time)
* mnist_net
* TSL16



* Attentions ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
* classes name must be contained in folder name 

## environment
My environment is based on 
* __ubuntu16__ 
* __cuda8__ (__cuda9.0__)
* __tensorflow_gpu1.4__ (__tensorflow_gpu1.10__ )
* __keras2.0.8__
* __numpy__
* __tqdm__
* __opencv-python__
* __scikit-learn__
### Install packages
* pip3 install tensorflow_gpu==1.4
* pip3 install keras==2.0.8
* pip3 install numpy
* pip3 install tqdm
* pip3 install opencv-python
* pip3 install scikit-learn

# 1.confirm config.py
* choose model and change parameter in config.py

# 2.train or test  dataset prepare
* python3 train.py

# 3.train your model
* __Train sigle model :__  python3 train.py modelName  (make sure the code use the args)
  check the input path and put your data in right place
* __Tensorboard :__ take LeNet as example, run " __tensorboard --logdir=./checkpoints/LeNet__ " to watch training with tensorboard


