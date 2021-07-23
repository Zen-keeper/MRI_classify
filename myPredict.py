"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just switch in config.py:
msg: You can choose the following model to train your image, and just switch in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,
    ResNet18,ResNet34,ResNet50,ResNet101,ResNet152,mnist_net
    TSL16
"""
from __future__ import print_function
from config import config
import sys,copy,shutil
from sklearn.metrics import confusion_matrix
import os,time,cv2

from keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import tensorflow as tf
import read_data
config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
tf.Session(config=config1)

from Build_model import Build_model

class PREDICT(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)

        try:
            className = sys.argv[2]
        except:
            print("use default className")
            className = "dog"

        # self.className = className
        # self.test_data_path = os.path.join(config.test_data_path,self.className)

    def classes_id(self):
        with open('train_class_idx.txt','r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        return lines

    def mkdir(self,path):
        if os.path.exists(path):
            return path
        os.mkdir(path)
        return path

    def calculate_metric(self,gt, pred):
        gt2 = []
        pred2 = []
        for i in range(len(pred)):
            pred2.append(0 if pred[i, 0] > pred[i, 1] else 1)
            gt2.append(0 if gt[i, 0] > gt[i, 1] else 1)
        confusion = confusion_matrix(gt2, pred2)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        auc = roc_auc_score(gt2, pred2)
        print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
        print('Sensitivity:', TP / float(TP + FN))
        print('Specificity:', TN / float(TN + FP))
        print('Auc:', auc)
        return (TP + TN) / float(TP + TN + FP + FN), TP / float(TP + FN), TN / float(TN + FP), auc

    def evaluate_model(self):
        print('\nTesting---------------')
        start = time.time()
        model = Build_model(self.config).build_model()
        weight_path = os.path.join(os.path.join(self.checkpoints, self.model_name+"_res"), self.model_name + '15.h5')
        if os.path.exists(weight_path):
            print('weights is loaded')
        else:
            print('weights is not exist')
        model.load_weights(weight_path)
        # loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)
        X_test, label_test, counter = read_data.process_img("D:\work\AD_V3\image_class\\ori\\AD", "D:\work\AD_V3\image_class\ori\\NC")
        pred1 = model.predict(X_test)
        accuracy, sensitivity, specificity ,auc = self.calculate_metric( label_test,pred1)
        # print('test loss;', loss)
        # print('test accuracy:', res)
        print("**************           accuracy: {}".format(accuracy))
        print("**************           sensitivity: {}".format(sensitivity))
        print("**************           specificity: {}".format(specificity))
        print("**************           auc: {}".format(auc))
        return accuracy, sensitivity, specificity,auc

    def Predict(self):
        start = time.time()
        model = Build_model(self.config).build_model()
        if os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'):
            print('weights is loaded')
        else:
            print('weights is not exist')
        model.load_weights(os.path.join(os.path.join(self.checkpoints,self.model_name),self.model_name+'.h5'))
        if(self.channles == 3):
            data_list = list(
                map(lambda x: cv2.resize(cv2.imread(os.path.join(self.test_data_path, x)),
                                         (self.normal_size, self.normal_size)), os.listdir(self.test_data_path)))
        elif(self.channles == 1):
            data_list = list(
                map(lambda x: cv2.resize(cv2.imread(os.path.join(self.test_data_path, x), 0),
                                         (self.normal_size, self.normal_size)), os.listdir(self.test_data_path)))
        img_list,label_lsit,counter = read_data.process_img("dataset/test/AD","dataset/test/NC")
        i,j,tmp = 0,0,[]
        for img in img_list:
            img = np.array([img_to_array(img)],dtype='float')/255.0
            pred = model.predict(img).tolist()[0]
            label = self.classes_id()[pred.index(max(pred))]
            confidence = max(pred)
            print('predict label     is: ',label)
            print('predict confidect is: ',confidence)

            if label != self.className:
                print('____________________wrong label____________________', label)
                i+=1
            else:
                j+=1

        accuracy = (1.0*j/ (1.0*len(data_list)))*100.0
        print("accuracy:{:.5}%".format(str(accuracy) ))
        print('Done')
        end = time.time()
        print("usg time:",end - start)

        with open("testLog/accuacy.txt","a") as f:
            f.write(config.model_name+","+self.className+","+"{:.5}%".format(str(accuracy))+"\n")

def main():
    mypredict = PREDICT(config)
    mypredict.evaluate_model()

if __name__=='__main__':
    main()
