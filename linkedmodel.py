import gc
from keras.preprocessing.image import img_to_array
import numpy as np
import sys,copy,shutil
from sklearn.metrics import confusion_matrix
import os,time,cv2
from keras.models import Sequential
import tensorflow as tf
import read_data
import random
from keras.layers import Dense, Dropout
from config import config
from keras.models import Model
config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
tf.Session(config=config1)

from Build_model import Build_model
def build_model(inputsize):
    model = Sequential()
    model.add(Dense(100, input_dim=inputsize, activation='sigmoid'))
    # model.add(Dropout(0.2))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

from sklearn.model_selection import train_test_split
class PREDICT(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)

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
            if (gt.ndim > 1):
                gt2.append(0 if gt[i, 0] > gt[i, 1] else 1)
        if (gt.ndim == 1):
            gt2 = gt
        confusion = confusion_matrix(gt2, pred2)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
        print('Sensitivity:', TP / float(TP + FN))
        print('Specificity:', TN / float(TN + FP))
        return (TP + TN) / float(TP + TN + FP + FN), TP / float(TP + FN), TN / float(TN + FP)

    def evaluate_model(self):
        print('\nTesting---------------')
        start = time.time()
        print('*'*10,'load ori model','*'*10)
        model_ori = Build_model(self.config).build_model()
        if os.path.join(os.path.join(self.checkpoints, self.model_name+"_ori"), self.model_name + '4.h5'):
            print('weights of ori model is loaded')
        else:
            print('weights is not exist')
        # model_ori.load_weights(os.path.join(os.path.join(self.checkpoints, self.model_name+"_ori"), self.model_name + '4.h5'))


        print('*' * 10, 'load residual model', '*' * 10)
        model_res = Build_model(self.config).build_model()
        if os.path.join(os.path.join(self.checkpoints, self.model_name + "_ori"), self.model_name + '3.h5'):
            print('weights of residual model is loaded')
        else:
            print('weights is not exist')
        # model_res.load_weights(
        #     os.path.join(os.path.join(self.checkpoints, self.model_name + "_res"), self.model_name + '4.h5'))
        # model = keras.layers.add(model_ori.get_layer('flatten_1'),model_res.get_layer('flatten_2'))
        feature_layer_ori_model = Model(inputs=model_ori.input,
                                        outputs=model_ori.get_layer('flatten_1').output)
        feature_layer_res_model = Model(inputs=model_res.input,
                                        outputs=model_res.get_layer('flatten_2').output)
        class_div_model = build_model(512)

        X_test_ori, label_ori, counter = read_data.process_img("D:/work/AD_V3/image_class/ori/AD", "D:/work/AD_V3/image_class/ori/NC")
        X_test_res, label_res, counter = read_data.process_img("D:/work/AD_V3/image_class/residual/AD", "D:/work/AD_V3/image_class/residual/NC")

        print(label_ori==label_res)

        pred_ori = feature_layer_ori_model.predict(X_test_ori)
        pred_res = feature_layer_res_model.predict(X_test_res)
        del feature_layer_res_model, feature_layer_ori_model
        print("一共释放了{}个对象".format(gc.collect()))

        input = np.add(pred_ori,pred_res)
        label = label_ori
        read_data.random_shuffle(input,label)

        train_x,test_x,train_y,test_y = train_test_split(input,label,test_size=0.2,random_state=random.randint(0, 100))


        history = class_div_model.fit(train_x, train_y,
                            validation_split=0.1,
                            epochs=500,
                            batch_size=120)
        score = class_div_model.evaluate(test_x, test_y)
        # pred1 = model_ori.predict(X_test)
        # accuracy, sensitivity, specificity = self.calculate_metric( label_test,pred1)
        # # print('test loss;', loss)
        # # print('test accuracy:', res)
        # print("**************           accuracy: {}".format(accuracy))
        # print("**************           sensitivity: {}".format(sensitivity))
        # print("**************           specificity: {}".format(specificity))
        # return accuracy, sensitivity, specificity


def main():
    mypredict = PREDICT(config)
    mypredict.evaluate_model()

if __name__=='__main__':
    main()