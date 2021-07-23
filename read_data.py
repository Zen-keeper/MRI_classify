# coding= utf-8
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nibabel as nib
import config

#输入一个文件路径，对其下的每个文件夹下的图片读取，并对每个文件夹给一个不同的Label
#返回一个img的list,返回一个对应label的list,返回一下有几个文件夹（有几种label)

def read_files(pathp,pathnegative,numofdata):
    if config.config.type == "ADNC2":
        print("the data type is ADNC")
        os_list = ["AD", "NC"]
    else:
        print("the data type is MCI")
        os_list = ["trans_", "NC"]
    # os_list = os.listdir(datapath)

    X = []
    Y = []
    n = numofdata
    for ptype in [pathp, pathnegative]:
        numofdata = n
        p = ptype
        # p = os.path.join(ptype, "ori")

        npy_paths = os.listdir(p)
        for npy_path in npy_paths:
            if npy_path.endswith(".hdr"):
                continue
            # npy_path2 = npy_path.split("mask")[1].split(".")[0]
            x_nii = readOnenii(p, npy_path)
            x_nii = np.expand_dims(x_nii, axis=3)
            X.append(x_nii)
            X.append(np.flip(x_nii,0))
            if ptype == pathnegative:
                Y.append(0)
                Y.append(0)
                print(0)
            else:
                Y.append(1)
                Y.append(1)
                print(1)
            numofdata = numofdata - 1
            if numofdata < 0:
                break
    # random_shuffle(X, Y)
    # random_shuffle(X, Y)
    print("数据总长度：" + str(len(X)))
    return np.array(X), np.array(Y)

def random_shuffle(data,label):
    randnum = np.random.randint(0, 1234)
    np.random.seed(randnum)
    np.random.shuffle(data)
    np.random.seed(randnum)
    np.random.shuffle(label)
    return data,label

def process_img(path1,path2,numofdata):
    x_paths = []
    y_labels = []
    image_paths = os.listdir(path1)
    n = numofdata
    for path_img in image_paths:
        if path_img.endswith(".hdr"):
            continue
        x_paths.append(path1 + "/" + path_img)
        y_labels.append(1)
        print(1)
        numofdata = numofdata-1
        if numofdata < 0:
            break
    image_paths = os.listdir(path2)
    numofdata = n
    for path_img in image_paths:
        if path_img.endswith(".hdr") or path_img.endswith("else_res.nii"):
            continue
        x_paths.append(path2 + "/" + path_img)
        y_labels.append(0)
        print(0)
        numofdata = numofdata - 1
        if numofdata < 0:
            break
    # random_shuffle(x_paths, y_labels)
    # random_shuffle(x_paths, y_labels)
    y_labels = np.array(y_labels)
    x_train_paths = x_paths
    y_train_paths = y_labels
    # x_train_paths, x_test_paths, y_train_paths, y_test_paths = train_test_split(x_paths, y_labels, test_size=0.1,
    #                                                               random_state=15)
    batch_res = []
    y_res = []
    for i in range(len(x_train_paths)):
        if(config.config.type!="combineT"):
            batch_res.append(read_single(x_train_paths[i],"path"))
            batch_res.append(read_singleflip(x_train_paths[i], "path"))
            batch_res.append(read_singleflip(x_train_paths[i], "path",1))
            # batch_res.append(read_singleflip(x_train_paths[i], "path",1))
        else:
            batch_res.append(read_combineT(x_train_paths[i],config.config.mask, "path"))
            batch_res.append(read_singleflip_combineT(x_train_paths[i], config.config.mask))
            batch_res.append(read_singleflip_combineT(x_train_paths[i], config.config.mask, 1))
            # batch_res.append(read_singleflip_combineT(x_train_paths[i], config.config.mask, 1))
        y_res.append(y_train_paths[i])
        y_res.append(y_train_paths[i])
        y_res.append(y_labels[i])
        # y_res.append(y_labels[i])
    y_res = np.array(y_res)
    # batch_test = []
    # y_test = []
    # for i in range(len(x_test_paths)):
    #     batch_test.append(read_single(x_test_paths[i],"path"))
    #     y_test.append(y_test_paths[i])
    #
    # y_test = np.array(y_test)
    return train_test_split(np.array(batch_res), y_res, test_size=0.1, random_state=15)
    # return np.array(batch_res), np.array(batch_test), y_res, y_test


def select_roi_th(path,threadhold):
    roi = nib.load(path).get_data()
    # c = "AD" if any(path.split("\\"))=="AD" else "NC"
    # print(path.split("\\")[5]+":"+str(np.max(roi)))
    mask = roi != roi
    mask[roi>threadhold] = True
    return mask

def readList(maskpath,roi_th,datapath,case=None):
    # mask = select_roi_count(maskpath,roi_count)
    if config.config.type == "AD":
        os_list = ["AD", "NC"]
    else:
        os_list = ["trans_", "NC"]
    X=[]
    Y=[]

    for class_name in os_list:
        p = os.path.join(datapath,class_name)
        npy_paths = os.listdir(p)
        for npy_path in npy_paths:
            if npy_path.endswith(".hdr"):
                continue
            if case is "residul":
                x_npy = np.load(os.path.join(p,npy_path))[mask] #套在残差图像上
                X.append(x_npy)
            else:
                mask_name = npy_path.split('mask')[1].split('.')[0]
                mask = select_roi_th(os.path.join(maskpath,class_name,mask_name+".nii"), roi_th)

                x_nii = nib.load(os.path.join(p,npy_path)).get_data()
                # 归一化
                img_fdata_flat = x_nii.flatten()
                img_fdata_flat = (img_fdata_flat - np.min(img_fdata_flat)) / (
                            max(img_fdata_flat) - min(img_fdata_flat))
                x_nii = np.reshape(img_fdata_flat, x_nii.shape)
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        for k in range(mask.shape[2]):
                            if (not mask[i, j, k]):
                                x_nii[i, j, k] = 0
                print(class_name)

                X.append(read_single(x_nii,"data"))
                    # for i in range(40, 90):
                    #     plt.imshow(x_nii[:, :, i], cmap='gray')
                    #     plt.waitforbuttonpress()
            if class_name.startswith("AD"):
                Y.append(1)
            else:
                Y.append(0)
    random_shuffle(X,Y)
    return np.array(X),np.array(Y)

def readOnenii(p, npy_path):
    x_nii = nib.load(os.path.join(p, npy_path)).get_data().astype(np.float32)
    # 归一化
    img_fdata_flat = x_nii.flatten()
    img_fdata_flat = (img_fdata_flat - np.min(img_fdata_flat)) / (
            max(img_fdata_flat) - min(img_fdata_flat))

    x_nii = np.reshape(img_fdata_flat, x_nii.shape)
    return x_nii

# def MultiReadList(residualpath,datapath,numofdata=500):
#     if config.config.type == "ADNC":
#         print("the data type is ADNC")
#         os_list = ["AD", "NC"]
#     else:
#         print("the data type is MCI")
#         os_list = ["trans_", "NC"]
#     # os_list = os.listdir(datapath)
#
#     X=[]
#     Y=[]
#     n=numofdata
#     for class_name in os_list:
#         numofdata = n
#         p = os.path.join(datapath,class_name)
#         p2 = os.path.join(residualpath,class_name)
#         npy_paths = os.listdir(p)
#         for npy_path in npy_paths:
#             if npy_path.endswith(".hdr"):
#                 continue
#             # npy_path2 = npy_path.split("mask")[1].split(".")[0]
#             npy_path2 = npy_path.split("_ori")[0]
#
#             x_nii = readOnenii(p, npy_path)
#             x_nii2 = readOnenii(p2,npy_path2+"_res.nii")
#
#             if(config.config.model_name=="ResNet3D"):
#                 temp1 = np.concatenate((x_nii, x_nii2), axis=0)
#                 temp1 = np.resize(temp1,(91,109,91))
#                 temp1 = np.expand_dims(temp1, axis=3)
#             else:
#                 temp1 = np.delete(read_single(x_nii, "data"), -1, axis=2)
#                 temp2 = np.delete(read_single(x_nii2, "data"), [1, 2], axis=2)
#                 temp1 = np.concatenate((temp1, temp2), axis=2)
#                 del temp2
#
#             X.append(temp1)
#             del temp1
#             # for i in range(40, 90):
#             #     plt.imshow(x_nii[:, :, i], cmap='gray')
#             #     plt.waitforbuttonpress()
#             if not class_name.startswith("NC"):
#                 Y.append(1)
#                 print(1)
#             else:
#                 Y.append(0)
#                 print(0)
#             numofdata = numofdata - 1
#             if numofdata < 0:
#                 break
#     random_shuffle(X,Y)
#     random_shuffle(X,Y)
#     print("数据总长度："+str(len(X)))
#     return np.array(X),np.array(Y)
#
#

def MultiReadList(pathp,pathnegative,numofdata=1000):
    if config.config.type == "ADNC2":
        print("the data type is ADNC")
        os_list = ["AD", "NC"]
    else:
        print("the data type is MCI")
        os_list = ["trans_", "NC"]
    x_paths = []
    y_labels = []
    image_paths = os.listdir(pathp)
    n = numofdata
    for path_img in image_paths:
        if path_img.endswith(".hdr"):
            continue
        x_paths.append(pathp + "/" + path_img)
        y_labels.append(1)
        print(1)
        numofdata = numofdata - 1
        if numofdata < 0:
            break
    image_paths = os.listdir(pathnegative)
    n = numofdata
    for path_img in image_paths:
        if path_img.endswith(".hdr"):
            continue
        x_paths.append(pathnegative + "/" + path_img)
        y_labels.append(1)
        print(1)
        numofdata = numofdata - 1
        if numofdata < 0:
            break
    x_train_paths, x_test_paths, y_train_paths, y_test_paths = train_test_split(x_paths, y_labels, test_size=0.2,
                                                                  random_state=15)
    X=[]
    Y=[]
    n=numofdata
    for ptype in [x_train_paths,pathnegative]:
        numofdata = n
        p = os.path.join(ptype,"ori")
        p2 = os.path.join(ptype,"residual")
        npy_paths = os.listdir(p)
        for npy_path in npy_paths:
            if npy_path.endswith(".hdr"):
                continue
            # npy_path2 = npy_path.split("mask")[1].split(".")[0]
            x_nii = readOnenii(p, npy_path)
            if(config.config.case=="combineT"):
                x_nii2 = readOneniiT(config.config.mask)
            else:
                npy_path2 = npy_path.split("_ori")[0]
                if(not os.path.isfile(os.path.join(p2, npy_path2+"_res.nii"))):
                    if(npy_path2[10]=="_"):
                        npy_path2 = npy_path2[0:10]+"-"+npy_path2[11:]
                    else:
                        npy_path2 = npy_path2[0:10] + "_" + npy_path2[11:]
                if (not os.path.isfile(os.path.join(p2, npy_path2 + "_res.nii"))):
                    continue
                x_nii2 = readOnenii(p2,npy_path2+"_res.nii")

            if(config.config.model_name=="ResNet3D"):
                temp1 = np.concatenate((x_nii, x_nii2), axis=0)
                temp1 = np.resize(temp1,(91,109,91))
                temp1 = np.expand_dims(temp1, axis=3)
                X.append(temp1)
                Y.append(0 if ptype==pathnegative else 1)
                x_nii = np.flip(x_nii,axis=0)
                x_nii2 = np.flip(x_nii2,axis=0)
                temp1 = np.concatenate((x_nii, x_nii2), axis=0)
                temp1 = np.resize(temp1, (91, 109, 91))
                temp1 = np.expand_dims(temp1, axis=3)
                X.append(temp1)
                Y.append(0 if ptype == pathnegative else 1)
            else:
                temp1 = np.delete(read_single(x_nii, "data"), -1, axis=2)
                temp2 = np.delete(read_single(x_nii2, "data"), [1, 2], axis=2)
                temp1 = np.concatenate((temp1, temp2), axis=2)
                X.append(temp1) #zhengchang
                # plt.imshow(np.concatenate((temp1[:,:,0], temp1[:,:,2])))
                # plt.waitforbuttonpress()
                temp1 = np.delete(read_singleflip(x_nii, "data",0), -1, axis=2)
                temp2 = np.delete(read_singleflip(x_nii2, "data", 0), [1, 2], axis=2)
                temp1 = np.concatenate((temp1, temp2), axis=2)
                # for ceng in range(3):
                #     plt.imshow(temp1[:,:,ceng])
                #     plt.waitforbuttonpress()
                X.append(temp1)#left-right
                temp1 = np.delete(read_singleflip(x_nii, "data", 1), -1, axis=2)
                temp2 = np.delete(read_singleflip(x_nii2, "data", 1), [1, 2], axis=2)
                temp1 = np.concatenate((temp1, temp2), axis=2)
                #
                X.append(temp1)
                temp1 = np.delete(read_singleflip(x_nii, "data", 1), -1, axis=2)
                temp2 = np.delete(read_singleflip(x_nii2, "data", 1), [1, 2], axis=2)
                temp1 = np.concatenate((temp1, temp2), axis=2)
                X.append(temp1)
                del temp1, temp2
                if ptype == pathnegative:
                    Y.append(0)
                    Y.append(0)
                    Y.append(0)
                    Y.append(0)
                    print(0)
                else:
                    Y.append(1)
                    Y.append(1)
                    Y.append(1)
                    Y.append(1)
                    print(1)



            numofdata = numofdata - 1
            if numofdata < 0:
                break
    random_shuffle(X,Y)
    # print("数据总长度："+str(len(X)))
    return np.array(X),np.array(Y)

import gaussianadded

def read_combineT(input,maskT,parmtype):
    # 主要作用是将三维数据转为二维切片
    if("path"==parmtype):
        img = nib.load(input).get_data().astype(np.float32)
        maskimg = nib.load(maskT).get_data().astype(np.float32)
        img[img < 0] = 0
    else:
        img = input

    img = np.transpose(img, [2, 1, 0])
    maskimg = np.transpose(maskimg, [2, 1, 0])
    wanted_img_size = config.config.normal_size
    # plt.imshow(img[40,:,:])
    # plt.waitforbuttonpress()
    # plt.imshow(maskimg[40,:,:])
    # plt.waitforbuttonpress()

    # 原始图像
    start = 10
    img_full_9 = np.array(img[start])
    for slice in range(1, 9):
        img_full_9 = np.concatenate((img_full_9, img[start + slice]), axis=1)
    start += 9
    img_full = img_full_9
    for hang in range(1,9):
        img_full_9 = np.array(img[start])
        for slice in range(1,9):
            img_full_9 = np.concatenate((img_full_9,img[start+slice]),axis = 1)
        start += 9
        img_full = np.concatenate((img_full,img_full_9),axis=0)
        del img_full_9
    # resize,增至2通道
    img_full = cv2.resize(np.array(img_full), (wanted_img_size, wanted_img_size))
    img_full = (img_full - np.min(img_full)) / (np.max(img_full) - np.min(img_full))
    img_full1 = np.expand_dims(img_full, axis=2)

    # mask
    start = 10
    img_full_9 = np.array(maskimg[start])
    for slice in range(1, 9):
        img_full_9 = np.concatenate((img_full_9, maskimg[start + slice]), axis=1)
    start += 9
    img_full = img_full_9
    for hang in range(1, 9):
        img_full_9 = np.array(img[start])
        for slice in range(1, 9):
            img_full_9 = np.concatenate((img_full_9, maskimg[start + slice]), axis=1)
        start += 9
        img_full = np.concatenate((img_full, img_full_9), axis=0)
        del img_full_9
    img_full = cv2.resize(np.array(img_full), (wanted_img_size, wanted_img_size))
    # img_full = (img_full - np.min(img_full)) / (np.max(img_full) - np.min(img_full))
    img_full = np.expand_dims(img_full, axis=2)
    img_full = np.concatenate((img_full1, img_full1, img_full), axis=2)
    # plt.imshow(np.array(img_full))
    # plt.waitforbuttonpress()
    return np.array(img_full)
def read_single(input,parmtype):
    # 主要作用是将三维数据转为二维切片
    if("path"==parmtype):
        img = nib.load(input).get_data().astype(np.float32)
        img[img < 0] = 0
        # 归一化
        # img_fdata_flat = img.flatten()
        # img_fdata_flat = (img_fdata_flat - np.min(img_fdata_flat)) / (max(img_fdata_flat) - min(img_fdata_flat))
        # img = np.reshape(img_fdata_flat, img.shape)
        # del img_fdata_flat
    else:
        img = input

    img = np.transpose(img, [2, 1, 0])
    wanted_img_size = config.config.normal_size

    # 定义初始的
    start = 10
    img_full_9 = np.array(img[start])
    for slice in range(1, 9):
        img_full_9 = np.concatenate((img_full_9, img[start + slice]), axis=1)
    start += 9
    img_full = img_full_9
    for hang in range(1,9):
        img_full_9 = np.array(img[start])
        for slice in range(1,9):
            img_full_9 = np.concatenate((img_full_9,img[start+slice]),axis = 1)
        start += 9
        img_full = np.concatenate((img_full,img_full_9),axis=0)
        del img_full_9
    # resize,增至三通道
    img_full = cv2.resize(np.array(img_full), (wanted_img_size, wanted_img_size))
    img_full = (img_full - np.min(img_full)) / (np.max(img_full) - np.min(img_full))
    img_full = np.expand_dims(img_full, axis=2)
    img_full = np.concatenate((img_full, img_full, img_full), axis=2)
    # plt.imshow(np.array(img_full))
    # plt.waitforbuttonpress()
    return np.array(img_full)
def read_singleflip(input,parmtype,flipdim = 0):
    # 主要作用是将三维数据转为二维切片
    if("path"==parmtype):
        img = nib.load(input).get_data().astype(np.float32)
        img[img < 0] = 0
        # 归一化
        # img_fdata_flat = img.flatten()
        # img_fdata_flat = (img_fdata_flat - np.min(img_fdata_flat)) / (max(img_fdata_flat) - min(img_fdata_flat))
        # img = np.reshape(img_fdata_flat, img.shape)
        # del img_fdata_flat
    else:
        img = input

    img = np.transpose(img, [2, 1, 0])
    # img = cv2.flip(img,0)
    if flipdim==0:
        img = np.flip(img, 0)
    else:
        # print("max:"+str(np.max(img)))
        img = gaussianadded.gaussian_noise(img, 0, np.max(img)/6)
    wanted_img_size = config.config.normal_size
    # 定义初始的
    start = 10
    img_full_9 = np.array(img[start])
    for slice in range(1, 9):
        img_full_9 = np.concatenate((img_full_9, img[start + slice]), axis=1)
    start += 9
    img_full = img_full_9
    for hang in range(1,9):
        img_full_9 = np.array(img[start])
        for slice in range(1,9):
            img_full_9 = np.concatenate((img_full_9,img[start+slice]),axis = 1)
        start += 9
        img_full = np.concatenate((img_full,img_full_9),axis=0)
        del img_full_9
    # resize,增至三通道
    img_full = cv2.resize(np.array(img_full), (wanted_img_size, wanted_img_size))
    img_full = (img_full - np.min(img_full)) / (np.max(img_full) - np.min(img_full))
    img_full = np.expand_dims(img_full, axis=2)
    img_full = np.concatenate((img_full, img_full, img_full), axis=2)
    # plt.imshow(np.array(img_full))
    # plt.waitforbuttonpress()

    return np.array(img_full)
def read_singleflip_combineT(input,maskT,flipdim = 0):
    # 主要作用是将三维数据转为二维切片

    img = nib.load(input).get_data().astype(np.float32)
    img[img < 0] = 0
    maskimg = nib.load(maskT).get_data().astype(np.float32)
    img = np.transpose(img, [2, 1, 0])
    maskimg = np.transpose(maskimg, [2, 1, 0])
    # img = cv2.flip(img,0)
    if flipdim==0:
        img = np.flip(img, 0)
    else:
        # print("max:"+str(np.max(img)))
        img = gaussianadded.gaussian_noise(img, 0, np.max(img)/5)
    wanted_img_size = config.config.normal_size
    # 原始图像
    start = 10
    img_full_9 = np.array(img[start])
    for slice in range(1, 9):
        img_full_9 = np.concatenate((img_full_9, img[start + slice]), axis=1)
    start += 9
    img_full = img_full_9
    for hang in range(1, 9):
        img_full_9 = np.array(img[start])
        for slice in range(1, 9):
            img_full_9 = np.concatenate((img_full_9, img[start + slice]), axis=1)
        start += 9
        img_full = np.concatenate((img_full, img_full_9), axis=0)
        del img_full_9
    # resize,增至2通道
    img_full = cv2.resize(np.array(img_full), (wanted_img_size, wanted_img_size))
    img_full = (img_full - np.min(img_full)) / (np.max(img_full) - np.min(img_full))
    img_full1 = np.expand_dims(img_full, axis=2)

    # mask
    start = 10
    img_full_9 = np.array(maskimg[start])
    for slice in range(1, 9):
        img_full_9 = np.concatenate((img_full_9, maskimg[start + slice]), axis=1)
    start += 9
    img_full = img_full_9
    for hang in range(1, 9):
        img_full_9 = np.array(img[start])
        for slice in range(1, 9):
            img_full_9 = np.concatenate((img_full_9, maskimg[start + slice]), axis=1)
        start += 9
        img_full = np.concatenate((img_full, img_full_9), axis=0)
        del img_full_9
    img_full = cv2.resize(np.array(img_full), (wanted_img_size, wanted_img_size))
    img_full = np.expand_dims(img_full, axis=2)
    img_full = np.concatenate((img_full1, img_full1, img_full), axis=2)
    return np.array(img_full)

#读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list



if __name__ == '__main__':
    # img_list,label_lsit,counter = process_img("D:/python_file/keras-resnet-master/Data/test/AD",
    #                                            "D:/python_file/keras-resnet-master/Data/test/NC")
    # print (counter)
    MultiReadList(r"D:\work\AD_V3\image_class\residual",r"D:\work\AD_V3\image_class\ori")
