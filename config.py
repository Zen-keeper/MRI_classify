# -*- coding: utf-8 -*-
"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just choose in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,
    ResNet18,ResNet34,ResNet50,ResNet101,ResNet152,mnist_net,TSL16
"""

import sys
class DefaultConfig():
    try:
        # model_name = sys.argv[1]
        # model_name = "AlexNet"AlexNet
        model_name = "ResNet18"
        # model_name = "LeNet"
    except:
        print("use default model VGG16, see config.py")
        model_name = "VGG16"

    print(model_name)
    # train_data_path = 'dataset/train'
    # test_data_path = 'dataset/test'

    type = "MCINC2"
    checkpoints = './checkpoints/' + type
    case = "combineT"
    nums0fmaxdata = 1000
    normal_size = 224
    epochs = 150
    batch_size = 16
    classNumber = 2 # see dataset
    channles = 3 # or 3 or 1
    lr = 0.005

    lr_reduce_patience = 8  # 需要降低学习率的训练步长
    early_stop_patience = 12 #提前终止训练的步长

    data_augmentation = False
    monitor = 'val_loss'
    cut = False
    rat = 0.01 #if cut,img[slice(h*self.rat,h-h*self.rat),slice(w*self.rat,w-w*self.rat)]
    mask = "E:\jscmask\\mci-nc_mask.nii"
    #mask = "E:\jscmask\\mcit_un_mask.nii"
    # mask = "E:\jscmask\\ad_nc_mask.nii"

config = DefaultConfig()

