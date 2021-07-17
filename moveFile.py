import os, shutil,glob
import numpy as np
import nibabel as nib
pathFrom = 'I:\\jscadnew'
pathTo = 'M:\AD\\ad_new1'
def test():
    p = "E:\jscmask\maskmci"
    fies = glob.glob(p+"\\*.img")
    for f in fies:
        img = nib.load(f).get_data().flatten()
        # if(max(img)==min(img)):
        if(np.isnan(np.array(img)).any()):
            print(f)
def move():
    objects = os.listdir(pathFrom)

    for i in range(len(objects)):
        object = objects[i]

        ori = os.listdir(os.path.join(pathFrom,object))[0]
        # resPath = glob.glob(pathFrom+'/'+object+'/residual/*gen(2).nii')
        # for p in resPath:
        temap = os.path.join(pathFrom,object,ori)
        # f = os.listdir(temp)[0]
        # shutil.copytree(os.path.join(temp,f),os.path.join(pathTo,object,f))
        # fs = glob.glob(temp+"\\*")
        # for p in fs:
        #     if os.path.isdir(p):
        #         f = str.split(p,"\\")[-1]
        #         shutil.copytree(p,os.path.join(pathTo,object,'residual',f))
            # tofile = str.split(p,"\\")[-2]
            # shutil.copy2(p,os.path.join(pathTo,object,'residual',tofile))
        if os.path.exists(os.path.join(pathTo,object,ori)):
            shutil.copytree(temp,os.path.join(pathTo,object,ori))
        # if not os.path.exists(os.path.join(pathTo,object,'residual')):
        #     continue
        # # t = glob.glob(temp+"\*(2).nii")[0]
        # # shutil.copy2(t,os.path.join(pathTo,object,'residual'))
        # for x in os.listdir(temp):
        #     if(x.endswith("(3).nii")):
        #     # if(os.path.isdir(os.path.join(temp,x))):
        #     #     files =glob.glob(os.path.join(pathFrom,object,'residual',x)+"*(3).nii")
        #     #     # for pp in os.listdir(os.path.join(pathFrom,object,'residual',x)):
        #     #     for pp in files:
        #     #         if not os.path.exists(pp):
        #         shutil.copy2(os.path.join(pathFrom,object,'residual',x),os.path.join(pathTo,object,'residual'))


def remove():
    pathFrom="F:\jsc_for_nc\\NC_V"
    objects = os.listdir(pathFrom)
    for i in range(65):
        p1 = os.path.join(pathFrom,objects[i],'residual')
        shutil.rmtree(p1)
        # for p in p1_s:
        # temp = os.path.join(p1, 'residual')
        # ps = os.listdir(temp)
        # for p2 in ps:
        #     p = os.path.join(temp,p2)
        #     if(not os.path.isfile(p)):
        #         shutil.rmtree(p)

def moveMCI():
    p = "K:\\Zhouhucheng\data_all\\ADNI_stableMCI_3T"
    to = "E:\\mcitrans\\mcistable"

    for f in os.listdir(p):
        p1 = os.path.join(p,f)
        for f2 in os.listdir(p1):
            if(f2.startswith("MP") or f2.startswith("Acc")):
                os.mkdir(os.path.join(to,f))
                shutil.copytree(os.path.join(p1,f2),os.path.join(to,f,f2))
move()