import platform

from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef


def calculate_metric(gt, pred):
    gt2 = []
    pred2 = []
    for i in range(len(pred)):
        pred2.append(0 if pred[i,0]>pred[i,1] else 1)
        gt2.append(0 if gt[i,0]>gt[i,1] else 1)
    confusion = confusion_matrix(gt2,pred2)
    mcc = matthews_corrcoef(gt2,pred2)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    auc = roc_auc_score(gt2, pred2)
    SEN = TP / float(TP+FN)
    PPV = TP/float(TP+FP)
    F_SCORE = (2*SEN*PPV)/(SEN+PPV)
    print('Accuracy:',(TP+TN)/float(TP+TN+FP+FN))
    print('Sensitivity:',SEN)
    print('Specificity:',TN / float(TN+FP))
    print('Auc:', auc)
    return (TP+TN)/float(TP+TN+FP+FN),TP / float(TP+FN),TN / float(TN+FP),auc,F_SCORE,mcc

def isWondows():
    '''
    判断当前运行平台
    :return:
    '''
    sysstr = platform.system()
    if (sysstr == "Windows"):
        return True
    elif (sysstr == "Linux"):
        return False
    else:
        print ("Other System ")
    return False