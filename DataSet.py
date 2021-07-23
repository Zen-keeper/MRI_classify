# -*- coding: utf-8 -*-


from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
import random

import utils
from read_data import *
import config





class DataSet(object):
    def __init__(self,path1,path2):
        self.num_classes = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.y_train = None
        self.y_test = None
        self.extract_data(path1,path2)
        #在这个类初始化的过程中读取path下的训练数据

    def extract_data(self,path1,path2):
        #根据指定路径读取出图片、标签和类别数
        # if(config.config.case=="ori" or config.config.case=="residual"):
        if(config.config.type=='ADNC3'):
            # path1 = r"F:/jsc_for_ad/forclass/AD/ad/"+str(config.config.case)
            path1 = r"F:/jsc_for_ad/forclass/AD/ad/ori"
            # path2 = r"E:/jsc_for_nc/NC/"+str(config.config.case)
            path2 = r"E:/jsc_for_nc/NC/ori"
        else:
            print("the data is Mci")
            if (utils.isWondows()):
                # path1 = r"E:/mcitrans/mcifor_c/mciall/" + str(config.config.case)
                # path2 = r"E:/jsc_for_nc/NC/" + str(config.config.case)
                path1 = r"E:/mcitrans/mcifor_c/mciall/ori"
                path2 = r"E:/jsc_for_nc/NC/ori"
            else:
                path1 = r"./dataset/mcifor_c/trans/" + str(config.config.case)
                path2 = r"./dataset/mcifor_c/untrans/" + str(config.config.case)

        # path1 = r"D:\work\AD_V3\image_class/" + str(config.config.case) + "/AD"
        # path2 = r"D:\work\AD_V3\image_class/" + str(config.config.case) + "/NC"
        if(config.config.model_name=="ResNet3D"):
            imgs, labels = read_files(path1,path2,config.config.nums0fmaxdata)
            counter = 2
        else:
            # imgs,labels, = process_img(path1,path2,config.config.nums0fmaxdata)
            # 现在返回的是图像路径
            X_train, X_test,self.y_train, self.y_test = process_img(path1,path2,config.config.nums0fmaxdata)
            counter = 2

        # readList 参数：maskpath是残差路径，下面应当有两个文件夹，与path2对应
        # 例： path1
        #        |AD
        #           |nii
        #           |nii
        #        |NC
        #           |nii
        #           |nii
        # else:
        #     imgs,labels = MultiReadList(path1,path2,config.config.nums0fmaxdata)
        #     counter = 2



        #将数据集打乱随机分组


        # X_train,X_test,self.y_train,self.y_test = train_test_split(imgs,labels,test_size=0.25,random_state=15)#random_state=random.randint(0, 100)
        # enhancedata(X_train,self.y_train)

        #重新格式化和标准化
        # 本案例是基于thano的，如果基于tensorflow的backend需要进行修改
        # X_train = X_train.reshape(X_train.shape[0], 256, 256, 3)
        # X_test = X_test.reshape(X_test.shape[0], 256, 256,3)


        # X_train = X_train.astype('float32')/255
        # X_test = X_test.astype('float32')/255
        print(X_train[1])

        #将labels转成 binary class matrices
        Y_train = np_utils.to_categorical(self.y_train, num_classes=counter)
        Y_test = np_utils.to_categorical(self.y_test, num_classes=counter)
        # print(Y_train)
        #将格式化后的数据赋值给类的属性上
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = counter

    def check(self):
        print('num of dim:', self.X_test.ndim)
        print('shape:', self.X_test.shape)
        print('size:', self.X_test.size)

        print('num of dim:', self.X_train.ndim)
        print('shape:', self.X_train.shape)
        print('size:', self.X_train.size)
        
if __name__ == '__main__':
    datast = DataSet(r"D:/python_file/keras-resnet-master/Data/test/AD",
                                               r"D:/python_file/keras-resnet-master/Data/test/NC")
    path='./dataset'
        