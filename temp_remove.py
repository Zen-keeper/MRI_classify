
import pandas as pd
import os,shutil
case = "ori"
path = "E:\mcitrans\mcifor_c/trans"
csvpath = "M:/MCI/MCI_trans_untrans.xls"

path_to = "E:\mcitrans\mcifor_c\\trans_remove"
df = pd.read_excel(csvpath,sheet_name="untrans")
data = df.values[:,0]
for i in range(len(data)):
    data[i]= data[i][0:10]
objects = os.listdir(path+"/"+case)
for i in range(len(objects)):
    if(objects[i][0:10]  in data):
        p1 = os.path.join(path,case, objects[i])
        if(not os.path.exists(path_to+"/"+case)):
            os.mkdir(path_to+"/"+case)
        shutil.move(p1,path_to+"/"+case)