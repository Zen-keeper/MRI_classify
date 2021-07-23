import  os,shutil
path = "I:\\NC_V2"
ps = os.listdir(path)
for object in ps:
    temp = os.path.join(path,object)
    intemps = os.listdir(temp)[0]
    temp = os.path.join(path, object,intemps)
    intemps = os.listdir(temp)
    for p in intemps:
        if(p not in ["label","mri","report"] and (not p.startswith("ADNI"))):
            os.remove(os.path.join(temp,p))