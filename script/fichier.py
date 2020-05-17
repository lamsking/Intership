# -*- coding: utf-8 -*-

import statsmodels as stat
import seaborn as sbrn
import pandas as pds
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")
import os, cv2, random
import numpy as np
import pandas as pd
%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import ticker
import seaborn as sns
%matplotlib inline 

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

#dataset = pds.read_csv('/home/inra-cirad/Bureau/MonDossier/output/dataset.csv')
#data = dataset.drop('WW', axis = 1)

#data.to_csv('/home/inra-cirad/Bureau/MonDossier/output/data.csv',index=False)
#data = dataset.drop(' ', axis = 1)



#data00=open('/home/inra-cirad/Bureau/MonDossier/output/data00.csv','w')
#data01=open('/home/inra-cirad/Bureau/MonDossier/output/data01.csv','w')
#data10=open('/home/inra-cirad/Bureau/MonDossier/output/data10.csv','w')
#data11=open('/home/inra-cirad/Bureau/MonDossier/output/data11.csv','w')
"""
dtset=open('/home/inra-cirad/Bureau/MonDossier/output/data.csv','r')
line=dtset.readlines()
for ln in line:
    l=ln.strip('\n').split(',')
    
    if l[1]=='0' and l[2]=='0':
        data00.write(ln)
    elif l[1]=='0' and l[2]=='1':
        data01.write(ln)
        
    elif l[1]=='1' and l[2]=='0':
        data10.write(ln)
    else:
        data11.write(ln)
"""       
        
#lecture des differentes base  
data00 = pds.read_csv('/home/inra-cirad/Bureau/MonDossier/output/out/data00.csv')
data01 = pds.read_csv('/home/inra-cirad/Bureau/MonDossier/output/out/data01.csv')
data10 = pds.read_csv('/home/inra-cirad/Bureau/MonDossier/output/out/data10.csv')
data11 = pds.read_csv('/home/inra-cirad/Bureau/MonDossier/output/out/data11.csv')
data00 = data00.iloc[:60,:]

#print(dataset_features["WD"].astype("str")+dataset_features["OW"].astype("str"))
#encodage des données

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#encodage de X

X00 = data00.iloc[:, :1]
#print(X00)
X01 = data01.iloc[:, :1]
X10 = data10.iloc[:, :1]
X11 = data11.iloc[:, :1]
X=np.concatenate((X00,X01,X10,X11))

#print(X)
#encodage de Y
data00['label'] = data00["WD"].astype("str")+data00["OW"].astype("str")
data01['label'] = data01["WD"].astype("str")+data01["OW"].astype("str")
data10['label'] = data10["WD"].astype("str")+data10["OW"].astype("str")
data11['label'] = data11["WD"].astype("str")+data11["OW"].astype("str")

Y00 = data00.iloc[:, 1:3].values
Y01 = data01.iloc[:, 1:3].values
Y10 = data10.iloc[:, 1:3].values
Y11 = data11.iloc[:, 1:3].values
#Y=str(Y00)+str(Y01)+str(Y10)+str(Y11)
Y=np.concatenate((Y00,Y01,Y10,Y11))
Y=np.concatenate((data00.label,data01.label,data10.label,data11.label))

print(X[0])
print(X)


#data split





#dataset_features["label"] = dataset_features["WD"].astype("str")+dataset_features["OW"].astype("str")
#datasetnew = dataset_features.drop('OW', axis = 1)
#dataset_labels = dataset.iloc[:, 3].values
#dataset_features.head(20)
#X = dataset_features.iloc[:, :1].values
print(X)
#Y = dataset_features.iloc[:,:4]
#Y = dataset.label

print(X[0][0])
import cv2
#im = cv2.imread(X[0][0],-1)
import os
os.chdir("/home/inra-cirad/Bureau/MonDossier")
#print(os.path.exists(X[0][0]))
#print(os.getcwdb())
Xs = []

for p in X:
    Xs.append(cv2.imread(p[0],-1))
Xs = np.array(Xs)
print(Xs)

from sklearn.model_selection import train_test_split
dataset_size = len(Xs)
Xs2dim = Xs.reshape(dataset_size,-1)
print(Xs2dim)
X_train, X_test, y_train, y_test = train_test_split(Xs2dim, Y, test_size=0.3, shuffle=False)






# preparing data by processing images using opencv
ROWS = 64
COLS = 64
CHANNELS = 3

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

train = prep_data(X_train, y_train)
test = prep_data(X_test,y_test)


#creation du model en utilisant le VGG16

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'


def wdow():
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS), activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))



    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model


model = wdow()



nb_epoch = 10
batch_size = 16
labs = train_data.iloc[:,1].values.tolist()

## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')        
        
history = LossHistory()
model.fit(train, labs, batch_size=batch_size, epochs=nb_epoch,
              validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])

'''
#prediction
predictions = model.predict(test, verbose=0)
predictions

loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()
plt.show()



for i in range(0,6):
    if predictions[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a Female'.format(predictions[i][0]))
    else: 
        print('I am {:.2%} sure this is a Male'.format(1-predictions[i][0]))
        
    plt.imshow(test[i].T)
    plt.show()'''






















#encodage des données
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
#classifier_knn =  KNeighborsClassifier(3) #classifier d'un KNN
#classifier_mlp = MLPClassifier(alpha=1, max_iter=1000) 
#classifier_rft = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
#classifier_assembliste = AdaBoostClassifier() #methode assembliste
classifier = svm.SVC(gamma=0.001)
dataset_size = len(Xs)
Xs2dim = Xs.reshape(dataset_size,-1)
print(Xs2dim)
X_train, X_test, y_train, y_test = train_test_split(Xs2dim, Y, test_size=0.5, shuffle=False)
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()


"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#data split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 ,random_state = 0)
"""