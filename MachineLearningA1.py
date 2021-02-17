
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from plot_metric.functions import BinaryClassification

### Read the target values and predicted scores into lists
def read_from_file(filename):
    file = open(filename, "r") 
    s = file.read()
    lines = s.split('\n')
    return [float(x) for x in lines]

def printConfusionMatrix(values):
    results = []
    for i in values:
        if i >= 0.25:
            results.append(1)
        else:
            results.append(0)
    return results,confusion_matrix(target,results)

def data(matrix,matrix2):
    precision = ((matrix[1][1])/(matrix[1][1] + matrix[0][1]))
    recall = ((matrix[1][1])/(matrix[1][1] + matrix[1][0]))
    f1 = ((2*(precision*recall)/(precision+recall)))
    precision2 = ((matrix2[1][1])/(matrix2[1][1] + matrix2[0][1]))
    recall2 = ((matrix2[1][1])/(matrix2[1][1] + matrix2[1][0]))
    f12 = ((2*(precision2*recall2)/(precision2+recall2)))
    print("Data for model 1:")
    ##TP/TP+FN
    print("TPR: ",matrix[1][1]/(matrix[1][1]+matrix[1][0]))
    #FP/FP+TN
    print("FPR: ",matrix[0][1]/(matrix[0][1]+matrix[0][0]))
    #TN/TN+FP
    print("TNR: ",matrix[0][0]/(matrix[0][0]+matrix[0][1]))
    #FN/FN+TP
    print("FNR: ",matrix[1][0]/(matrix[1][0]+matrix[1][1]))
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Measure: ", f1)
    print(" ")
    print("------------------------------------")
    print(" ")
    print("Data for model 2: ")
    print("TPR: ",matrix[1][1]/(matrix[1][1]+matrix[1][0]))
    print("FPR: ",matrix[0][1]/(matrix[0][1]+matrix[0][0]))
    print("TNR: ",matrix[0][0]/(matrix[0][0]+matrix[0][1]))
    print("FNR: ",matrix[1][0]/(matrix[1][0]+matrix[1][1]))
    print("Precision: ", precision2 )
    print("Recall: ",recall2)
    print("F1-Measure: ",f12)
    
    
def drawCurve(results,target,TPR,FPR):
    #y_true = target
    #y_pred = results
    
        #this also works 
    #fpr,tpr,thresholds = metrics.roc_curve(target,results,pos_label=0)
    #plt.plot(tpr,fpr)
    #plt.show()
    
    #this also works 
    
    #fpr,tpr,thresholds = metrics.roc_curve(target,results,pos_label=0)
    #plt.plot(tpr,fpr)
    #plt.show()
    
    bc = BinaryClassification(target,results,labels=["Class1","Class2"],threshold=0.25)
    plt.figure(figsize=(5,5))
    
    bc.plot_roc_curve()
    plt.show()

#the roc index for both 
def AUC(pred):
    fpr,tpr,thresholds = metrics.roc_curve(target,pred,pos_label=0)
    print(metrics.auc(tpr,fpr))
    auc = np.trapz(tpr,fpr)
    print("AUC: ",auc)
    
def k_test(pred):
    print(stats.kstest(pred,'norm'))
    return stats.kstest(pred,'norm')

target = read_from_file('target.csv')
pred_1 = read_from_file('pred_model1.csv')
pred_2 = read_from_file('pred_model2.csv')

results, matrix = printConfusionMatrix(pred_1)
print(matrix)
print("------------------")
results2, matrix2 = printConfusionMatrix(pred_2)
print(matrix2)

data(matrix,matrix2)

print("ROC Curve for Model 1: ")
drawCurve(pred_1, target,matrix[0][0],matrix[0][1])
print(" ")
print("--------------------------------------------")
print(" ")
print("ROC Curve for Model 2: ")
drawCurve(pred_2,target, matrix2[0][0],matrix2[0][1])

print("Model 1:")
AUC(pred_1)
print("Model 2: ")
AUC(pred_2)

print(" ")
print("--------------------------------------------")
print(" ")

print("K-S Stats for model 1: ")
k1 = k_test(pred_1)
print("K-S Stats for model 2: ")
k2 = k_test(pred_2)

if k1 > k2:
    print("According to the K-S Statistic model 1 has better preformance: ",k1)
else:
    print("According to the K-S Statistic model 2 has better preformance: ",k2)























