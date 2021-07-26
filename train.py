"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just switch in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,esNet18,ResNet34,ResNet50,ResNet_101,ResNet_152
"""

from __future__ import print_function

import gc
import xlwt
from config import config
import numpy as np
import os,glob,itertools,tqdm,cv2,keras
from random import shuffle

from keras.preprocessing.image import img_to_array,ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import DataSet
import test_gpu

test_gpu.set_gpu()

import tensorflow as tf
config1 = tf.ConfigProto(log_device_placement=True)
config1.gpu_options.allow_growth = True
tf.Session(config=config1)

import sys
sys.setrecursionlimit(10000)

from Build_model import Build_model

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import utils
import newtensorboard
class Train(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)
        self.accuracy_max = 0
        self.file = xlwt.Workbook()
        self.sheet = self.file.add_sheet('test', cell_overwrite_ok=True)
        self.sheet.write(0,0,str(config.model_name))
        self.sheet.write(0,1,"normal_size"+str(config.normal_size))
        self.sheet.write(0,2,"batchsize"+str(config.batch_size))
        self.sheet.write(0,3,"lr"+str(config.lr))



    def try_on_test(self,model, X_test, y_test,k):
        pred = model.predict(np.array(X_test))
        accuracy, sensitivity, specificity ,auc,F_SCORE,mcc = utils.calculate_metric(y_test, pred)

        # self.file.write("accuracy:" + str(k) + str(accuracy) + "\n")
        # self.file.write("sensitivity:" + str(k) + str(sensitivity) + "\n")
        # self.file.write("specificity:" + str(k) + str(specificity) + "\n")
        self.sheet.write(k, 1, str(accuracy))
        self.sheet.write(k, 2, str(sensitivity))
        self.sheet.write(k, 3, str(specificity))
        self.sheet.write(k, 4, str(auc))
        self.sheet.write(k, 5, str(F_SCORE))
        self.sheet.write(k, 6, str(mcc))
        if (self.accuracy_max < accuracy):
            model.save('./my_model.h5')
            self.accuracy_max = accuracy
    def try_on_v(self,model, X_test, y_test,k):
        pred = model.predict(np.array(X_test))
        accuracy, sensitivity, specificity ,auc,FCORE,MCC = utils.calculate_metric(y_test, pred)

        # self.file.write("accuracy:" + str(k) + str(accuracy) + "\n")
        # self.file.write("sensitivity:" + str(k) + str(sensitivity) + "\n")
        # self.file.write("specificity:" + str(k) + str(specificity) + "\n")
        self.sheet.write(k, 8, str(accuracy))
        self.sheet.write(k, 9, str(sensitivity))
        self.sheet.write(k, 10, str(specificity))
        self.sheet.write(k, 11, str(auc))
        self.sheet.write(k, 12, str(FCORE))
        self.sheet.write(k, 13, str(MCC))
        # if (self.accuracy_max < accuracy):
        #     model.save('./my_model.h5')
        #     self.accuracy_max = accuracy

    def get_file(self,path):
        ends = os.listdir(path)[0].split('.')[-1]
        img_list = glob.glob(os.path.join(path , '*.'+ends))
        return img_list

    # def load_data(self):
    #
    #     categories = list(map(self.get_file, list(map(lambda x: os.path.join(self.train_data_path, x), os.listdir(self.train_data_path)))))
    #     data_list = list(itertools.chain.from_iterable(categories))
    #     shuffle(data_list)
    #     images_data ,labels_idx,labels= [],[],[]
    #
    #     with_platform = os.name
    #
    #     for file in tqdm.tqdm(data_list):
    #         if self.channles == 3:
    #             img = cv2.imread(file)
    #             # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #             # img = cv2.threshold(img,128,255,cv2.THRESH_BINARY)[-1]
    #             _, w, h = img.shape[::-1]
    #         elif self.channles == 1:
    #             # img=cv2.threshold(cv2.imread(file,0), 128, 255, cv2.THRESH_BINARY)[-1]
    #             img = cv2.imread(file,0)
    #             # img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[-1]
    #             w, h = img.shape[::-1]
    #
    #         if self.cut:
    #             img = img[slice(int(h*self.rat),int(h-h*self.rat)),slice( int(w*self.rat),int(w-w*self.rat) )]
    #         img = cv2.resize(img,(self.normal_size,self.normal_size))
    #         if with_platform == 'posix':
    #             label = file.split('/')[-2]
    #         elif with_platform=='nt':
    #             label = file.split('\\')[-2]
    #
    #         # print('img:',file,' has label:',label)
    #         img = img_to_array(img)
    #         images_data.append(img)
    #         labels.append(label)
    #
    #     with open('train_class_idx.txt','r') as f:
    #         lines = f.readlines()
    #         lines = [line.rstrip() for line in lines]
    #         for label in labels:
    #             idx = lines.index(label.rstrip())
    #             labels_idx.append(idx)
    #
    #     # images_data = np.array(images_data,dtype='float')/255.0
    #     # images_data = np.array(images_data, dtype='float32') / 255.0
    #     labels = to_categorical(np.array(labels_idx),num_classes=self.classNumber)
    #     X_train, X_test, y_train, y_test = train_test_split(images_data,labels)
    #     return X_train, X_test, y_train, y_test

    def mkdir(self,path):
        if not os.path.exists(path):
            return os.mkdir(path)
        return path

    def train(self,X_train, X_test, y_train, y_test,model,k):
        print("*"*50)
        print("-"*20+"train",config.model_name+"-"*20)
        print("*"*50)
        # gpus = os.environ["CUDA_VISIBLE_DEVICES"]


        # tensorboard=TensorBoard(log_dir=self.mkdir(os.path.join(self.checkpoints,self.model_name,config.case) ),batch_size=1)
        tensorboard= newtensorboard.TrainValTensorBoard(log_dir=self.mkdir(os.path.join(self.checkpoints, self.model_name, config.case)))

        lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor=config.monitor,
                                                      factor=0.5,
                                                      patience=config.lr_reduce_patience,
                                                      verbose=1,
                                                      mode='auto',
                                                      cooldown=0)
        early_stop = keras.callbacks.EarlyStopping(monitor=config.monitor,
                                                   min_delta=0,
                                                   patience=config.early_stop_patience,
                                                   verbose=1,
                                                   mode='auto')
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(self.mkdir( os.path.join(self.checkpoints,self.model_name,config.case) ),self.model_name+str(k)+'.h5'),
                                                     monitor=config.monitor,
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='auto',
                                                     period=1)

        if self.data_augmentation:
            print("using data augmentation method")
            data_aug = ImageDataGenerator(
                rotation_range=5,  # 图像旋转的角度
                width_shift_range=0.2,  # 左右平移参数
                height_shift_range=0.2,  # 上下平移参数
                zoom_range=0.3,  # 随机放大或者缩小
                horizontal_flip=True,  # 随机翻转
            )

            data_aug.fit(X_train)

            model.fit_generator(
                data_aug.flow(X_train, y_train, batch_size=config.batch_size),
                steps_per_epoch=X_train.shape[0] // self.batch_size,
                validation_data=(X_test, y_test),
                epochs=self.epochs, verbose=1, max_queue_size=1000,
                callbacks=[lr_reduce,checkpoint,tensorboard],#early_stop,
            )
        else:

            model.fit(x=X_train,y=y_train,
                      batch_size=self.batch_size,
                      validation_data=(X_test,y_test),
                      epochs=self.epochs,
                      callbacks= [checkpoint,early_stop,lr_reduce,tensorboard],#early_stop
                      shuffle=True,
                      verbose=1)

    def start_train(self):
        from sklearn.model_selection import StratifiedKFold

        if(utils.isWondows()):
            dataset = DataSet.DataSet(r"F:/jsc_for_ad/forclass/AD/ad/",
                                      r"E:\jsc_for_nc/NC/")#r"E:/jsc_for_nc/NC/""F:\jsc_for_ad\forclass\AD\ad\\"E:/mcitrans\mcifor_c\untrans

        else:
            dataset = DataSet.DataSet(r"./dataset/mcifor_c/trans",
                                      r"./dataset/mcifor_c/untrans")
        gc.collect()
        skf = StratifiedKFold(10)
        # 训练集测试集路径划分
        X_train_p, X_test_p, y_train, y_test = dataset.X_train,dataset.X_test,dataset.Y_train,dataset.Y_test
        X_test_p
        # 读取测试集数据
        X_test = []
        YY_test = []
        for i in range(len(X_test_p)):
            if (config.case == "combineT"):
                X_test.append(DataSet.read_combineT(X_test_p[i], config.mask,"path"))

                # X_test.append(DataSet.read_singleflip_combineT(X_test_p[i], config.mask))
                # X_test.append(DataSet.read_singleflip_combineT(X_test_p[i], config.mask,1))
                YY_test.append(y_test[i])
                # YY_test.append(y_test[i])
                # YY_test.append(y_test[i])
            else:
                X_test.append(DataSet.read_single(X_test_p[i], "path"))
                # X_test.append(DataSet.read_singleflip(X_test_p[i], "path"))
                X_test.append(DataSet.read_singleflip(X_test_p[i], "path",1))
                # YY_test.append(y_test[i])
                # YY_test.append(y_test[i])
                YY_test.append(y_test[i])

        X_test = np.array(X_test)
        y_test = np.array(YY_test)
        # y_test = (np.arange(2) == y_test[:, None]).astype(int)
        k = 1
        # K折交叉验证
        for cnt, (train, valid) in enumerate(skf.split(X_train_p,dataset.y_train)):
            if(config.case!="combineT"):
                X_train = []
                y_res = []
                for i in train:
                    X_train.append(DataSet.read_single(X_train_p[i], "path"))
                    X_train.append(DataSet.read_singleflip(X_train_p[i], "path"))
                    X_train.append(DataSet.read_singleflip(X_train_p[i], "path",1))
                    X_train.append(DataSet.read_singleflip(X_train_p[i], "path",1))

                    y_res.append(dataset.y_train[i])
                    y_res.append(dataset.y_train[i])
                    y_res.append(dataset.y_train[i])
                    y_res.append(dataset.y_train[i])
                X_train = np.array(X_train)
                y_res = np.array(y_res)
                y_res = (np.arange(2) == y_res[:, None]).astype(int)
                X_valid = []
                y_test_res = []
                for i in valid:
                    X_valid.append(DataSet.read_single(X_train_p[i], "path"))
                    X_valid.append(DataSet.read_singleflip(X_train_p[i], "path"))
                    y_test_res.append(y_train[i])
                    y_test_res.append(y_train[i])
                X_test = np.array(X_test)
                y_test_res = np.array(y_test_res)
                # y = (np.arange(2) == y_res[:, None]).astype(int)
            else:
                X_train = []
                y_res = []
                for i in train:
                    X_train.append(DataSet.read_combineT(X_train_p[i],config.mask, "path"))
                    X_train.append(DataSet.read_singleflip_combineT(X_train_p[i],config.mask))
                    X_train.append(DataSet.read_singleflip_combineT(X_train_p[i],config.mask, 1))
                    # X_train.append(DataSet.read_singleflip_combineT(X_train_p[i],config.mask, 1))
                    #
                    # y_res.append(dataset.y_train[i])
                    y_res.append(dataset.y_train[i])
                    y_res.append(dataset.y_train[i])
                    y_res.append(dataset.y_train[i])
                X_train = np.array(X_train)
                y_res = np.array(y_res)
                y_res = (np.arange(2) == y_res[:, None]).astype(int)
                X_valid = []
                y_test_res = []
                for i in valid:
                    X_valid.append(DataSet.read_combineT(X_train_p[i], config.mask,"path"))
                    X_valid.append(DataSet.read_singleflip_combineT(X_train_p[i], config.mask))
                    y_test_res.append(y_train[i])
                    y_test_res.append(y_train[i])
                X_test = np.array(X_test)
                y_test_res = np.array(y_test_res)
                # y = (np.arange(2) == y_res[:, None]).astype(int)
            for count in range(1):
                model = Build_model(config).build_model()
                self.train(np.array(X_train), np.array(X_valid), y_res, y_test_res,model,k)
                self.try_on_v(model,X_valid,y_test_res,k)
                self.try_on_test(model,X_test,y_test,k)
                del model
                gc.collect()
                k = k + 1
                self.file.save("{}/{}test_{}_{}.xls".format(config.checkpoints,config.type ,config.model_name,config.case))

        # # 单次训练
        # model = Build_model(config).build_model()
        # self.train(X_train, X_test, y_train, y_test, model, k)
        # self.try_on_test(model, X_test, y_test, k)
        # self.file.save("./test.xls")

    def remove_logdir(self):
        self.mkdir(self.checkpoints)
        self.mkdir(os.path.join(self.checkpoints,self.model_name))
        events = os.listdir(os.path.join(self.checkpoints,self.model_name))
        for evs in events:
            if "events" in evs:
                os.remove(os.path.join(os.path.join(self.checkpoints,self.model_name),evs))
        # if(os.path.exists("./test.txt")):
        #     os.remove("./test.txt")

    def mkdir(self,path):
        if os.path.exists(path):
            return path
        os.mkdir(path)
        return path


def main():
    train = Train(config)
    train.remove_logdir()
    train.start_train()
    print('Done')

if __name__=='__main__':
    main()
