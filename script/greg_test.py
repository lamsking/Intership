#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:49:07 2019

@author: pam
"""

# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os, cv2, random
import numpy as np
import pandas as pd
#%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import ticker
import seaborn as sns
#%matplotlib inline 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, SpatialDropout2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
import os
import sys
#from tensorflow.keras import backend


# loading labels for each image from csv
data = pd.read_csv('data.csv')
labels = data.iloc[:,0:2]
########################
labels_1 = labels[labels['WD']==1]

labels_0 = labels[labels['WD']==0][:200]
f = [labels_1,labels_0]
######################################################
os.chdir("//home/inra-cirad/Bureau/MonDossier/")

X= labels.iloc[:, :1].values
y = labels.iloc[:, 1:3].values
# print(len(y))

Xs = []

print(cv2.imread(X[0][0],-1).shape)

for p in X:
    Xs.append(cv2.imread(p[0],-1))
Xs = np.array(Xs)

# print(len(p))
#dataset_size = len(Xs)
# z = np.random.permutation(len(Xs))
# Xs = Xs[z]
#k = np.random.permutation(len(y))
# y = y[z]


#Xs2dim = Xs.reshape(dataset_size,-1)
# print(Xs)

Xs_train, Xs_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, shuffle=True)
Xs_train = Xs_train.reshape(Xs_train.shape[0],1,256,320)
Xs_test = Xs_test.reshape(Xs_test.shape[0],1,256,320)
#train_data = pd.concat([X_train, y_train]).drop_duplicates(keep=False)
#test_data=pd.concat([X_test,y_test]).drop_duplicates(keep=False)

# print(y_test)
unique, counts = np.unique(y_test, return_counts=True)
# print(dict(zip(unique, counts)))
# sys.exit(0)
'''
# Separating WD labels
wd_data = labels[labels['WD'] == 1]
print(wd_data)
wd_data.head()
# Splitting WD data into train and test
test_wd_data = wd_data.iloc[-4:,:]
train_wd_data = wd_data.iloc[:-4,:]
print(len(test_wd_data))

# Separating female labels
other_data = labels[labels['WD'] == 0]
other_data.head()
'''

'''
# Splitting male data into train and test
test_other_data = other_data.iloc[-16:,:]
train_other_data = other_data.iloc[:-16,:]

# total test data
test_indices = test_other_data.index.tolist() + test_wd_data.index.tolist()
test_data = labels.iloc[test_indices,:]
test_data.head()

# total train data
train_data = pd.concat([labels, test_data, test_data]).drop_duplicates(keep=False)
train_data.head()
'''
# train and test with image name along with paths
path = ''

#train_image_name = [path+each for each in train_data['image'].values.tolist()]
#test_image_name = [path+each for each in test_data['image'].values.tolist()]


# preparing data by processing images using opencv
ROWS =256
COLS = 320
CHANNELS = 1

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%5 == 0: print('Processed {} of {}'.format(i, count))
        
    return data
'''
train = prep_data(train_image_name)
test = prep_data(test_image_name)

# checking count of male and females
sns.countplot(labels['WD'])

# plotting female and male side by side
def show_wd_and_other():
    other = read_image(train_image_name[0])
    wd = read_image(train_image_name[2])
    pair = np.concatenate((other, wd), axis=1)
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    plt.show()
    
show_wd_and_other()

# splitting path of all images 
train_wd_image = []
train_other_image = []
for each in train_image_name:
    if each in train_wd_data['image'].values:
        train_wd_image.append(each)
    else:
        train_other_image.append(each)
'''
#Creating VGG 16 model for training 
# optimizer = RMSprop(lr=1e-4)
#optimizer = 'adam'
# objective = 'binary_crossentropy'


def wd_other():
    
    model = Sequential()

    model.add(SpatialDropout2D(0.2, input_shape=(CHANNELS, ROWS, COLS)))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    # model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    # model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    # model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    # model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
        
    # model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    # model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    # #model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))#enlever
    # model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))



    model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    
    # model.add(Dense(256, activation='softmax'))
    model.add(Dropout(0.5))

    model.add(Dense(18, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    # model.add(Activation('sigmoid'))
    #model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    return model


model = wd_other()

model.summary()

#nb_epoch = 500
#batch_size = 4
#labs = labels.iloc[:,1].values.tolist()


#print(Xs2dim)



## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, verbose=0, epsilon=0.5e-5, mode='min')
earlyStopping = EarlyStopping(monitor='val_loss', patience=500, verbose=0, mode='auto') 
# early_stopping = EarlyStopping(monitor='val_loss', patience=250, verbose=1, mode='auto')        
        
history = LossHistory()


print(y_train)


model.fit(Xs_train,y_train,validation_data=(Xs_test,y_test), epochs=10, batch_size=500, 
            verbose=2, shuffle=True, callbacks=[history, EarlyStopping])

#########
#prediction et matrice de confusion
predictions = model.predict(Xs_test, verbose=1)
predict = model.predict(Xs.reshape(Xs.shape[0],1,256,320))
np.array(np.c_[predict,y])
np.savetxt("test_classif_test.csv", np.c_[predict,y], delimiter=";")
# print(predictions)
# for i in range(0,len(y_test)):
#     print(predictions[i][0],' = ',y_test[i][0])
    
loss = history.losses
val_loss = history.val_losses



# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('VGG-16 Loss Trend')
# plt.plot(loss, 'blue', label='Training Loss')
# plt.plot(val_loss, 'green', label='Validation Loss')
# plt.xticks(range(0,10)[0::2])
# plt.legend()
# plt.show()

# import seaborn as sn
# y_pred = []
# y_actuel = []
# mat_conf = []
# mat_conf.append(y_test)
# #print(len(predictions))
# for i in range(0,len(y_test)):
#      y_actuel.append(y_test[i][0])
#      if predictions[i, 0] >= 0.4450765:    
#          print('It is {:.7%} sure this is a WD'.format(predictions[i][0]))
#          y_pred.append(1)
         
       
#      else: 
#          print('It is {:.7%} sure this is a other'.format(1-predictions[i][0]))
#          y_pred.append(0)

# data = {'yactuel':y_actuel,'ypred':y_pred}
# #print(data)
# df_mat = pd.DataFrame(data,columns = ['yactuel','ypred'])

# confusion_mat = pd.crosstab(df_mat['yactuel'],df_mat['ypred'],rownames=['Actuel'],colnames=['Predictiion'])
# print(confusion_mat)
# sn.heatmap(confusion_mat,annot=True)


# #######################################

# #enregistement des poids et du model

# model.save_weights('/home/pam/apprentissage/model/lundi08/wd_5000_other.h5')
# model.save('/home/pam/apprentissage/model/lundi08/model_5000.model')




#predictions = model.predict(test, verbose=0)
#print(predictions)

#loss = history.losses
#val_loss = history.val_losses

# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('VGG-16 Loss Trend')
# plt.plot(loss, 'blue', label='Training Loss')
# plt.plot(val_loss, 'green', label='Validation Loss')
# plt.xticks(range(0,nb_epoch)[0::2])
# plt.legend()
# plt.show()


# for i in range(0,12):
#     if predictions[i, 0] >= 0.5: 
#         print('It is {:.2%} sure this is a other'.format(predictions[i][0]))
#     else: 
#         print('It is {:.2%} sure this is a wd'.format(1-predictions[i][0]))
#     print(test_image_name[i])    
#     plt.imshow(test[i].T)
#     plt.show()



