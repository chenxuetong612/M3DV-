from numpy import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model

my_model=load_model('./最优神经网络.h5')
BATCH_SIZE=128
NUM_CLASSES=2
NUM_EPOCHS=15
adress1='C:/Users/lenovo/Desktop/sjtu-m3dv-medical-3d-voxel-classification/train_val'
adress2='C:/Users/lenovo/Desktop/sjtu-m3dv-medical-3d-voxel-classification'
adress3='C:/Users/lenovo/Desktop/sjtu-m3dv-medical-3d-voxel-classification/test'
import os
x1=np.load(adress1+'/candidate'+str(1)+'.npz')
x_train=[x1 for p in range(465)]
x_trainvoxel=[x1['voxel'] for p in range(465)]
x_trainseg=[x1['seg'] for p in range(465)]
j=0
i=0
while (i<584) :
    if(os.path.exists(adress1+'/candidate'+str(i)+'.npz')):
        x1=np.load(adress1+'/candidate'+str(i)+'.npz')
        x_train[j]=x1
        x_trainvoxel[j]=x_train[j]['voxel']
        x_trainseg[j]=x_train[j]['seg']
        j+=1
    i+=1
x_train=x_trainvoxel*(np.array(x_trainseg).astype(int))
np.shape(x_train)
y_train1=pd.read_csv(adress2+'/train_val.csv')
y_train=[1 for p in range(465)]
for z in range(200):
    y_train[z]=y_train1['lable'][z]
y_train1=y_train
import os
x2=np.load(adress3+'/candidate'+str(11)+'.npz')
x_test=[x2 for p in range(117)]
x_testvoxel=[x2['voxel'] for p in range(117)]
x_testseg=[x2['seg'] for p in range(117)]
y_testname=[' ' for p in range(117)]
j=0
i=0
while (i<583) :
    if(os.path.exists(adress3+'/candidate'+str(i)+'.npz')):
        x2=np.load(adress3+'/candidate'+str(i)+'.npz')
        x_test[j]=x2
        x_testvoxel[j]=x_test[j]['voxel']
        x_testseg[j]=x_test[j]['seg']
        y_testname[j]='candidate'+str(i)
        j+=1
    i+=1
x_test=x_testvoxel*(np.array(x_testseg).astype(int))
np.shape(x_test)
x_train=x_train.astype('float32')/255.
x_test=x_test.astype('float32')/255.
y_train=keras.utils.to_categorical(y_train, NUM_CLASSES)
input_shape=(100, 100, 100)
a=my_model.predict(x_test)
b=a[:,1]
dataframe=pd.DataFrame({'Id':y_testname,'Predicted':b})
dataframe.to_csv(adress2+'/Submission.csv',index=False,sep=',')
from sklearn.metrics import roc_auc_score
c=my_model.predict(x_train)
d=[0 for p in range(465)]
for z in range(465):
    if y_train1[z]==0:
        d[z]=c[z][0]
    else:
        d[z]=c[z][1]
roc_auc_score(y_train1,d)
