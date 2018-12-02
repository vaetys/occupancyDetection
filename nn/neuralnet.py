import os,sys
import random
import numpy as np
import tensorflow as tf
import filecmp
from keras.models import Model
from keras.layers import Input, Dense
from keras import optimizers
from time import strptime, strftime, mktime, gmtime
import random
from sklearn import preprocessing


def datetime_converter(datetime_string):
    # (1) Convert to datetime format
    target_timestamp = strptime(datetime_string, '%Y-%m-%d %H:%M:%S')

    # (2) mktime creates epoch time from the local time
    mktime_epoch = mktime(target_timestamp)
    #print(int(mktime_epoch)) # convert to integer to remove decimal

    # (3) gmtime to convert epoch time into UTC time object
    epoch_to_timestamp = strftime('%Y-%m-%d %H:%M:%S', gmtime(mktime_epoch))
    #print(epoch_to_timestamp)
    return int(mktime_epoch)

#----------------------------------------------------------------
trainDataPath = '../data/occupancy_data/datatrainingRan.txt'
testDataPath = '../data/occupancy_data/datatestRan.txt'
testData2Path = '../data/occupancy_data/datatest2.txt'
#----------------------------------------------------------------

def loadData(path):
    data = np.genfromtxt(path, dtype=float, delimiter=',', skip_header=0, encoding=None, usecols = (1, 2, 3, 4, 5))
    data[:, 0] = getTimes(path)
    data = normalize(data)
    return data

#Returns a spearate vector out of the last column of the given 2 dimensional numpy array.
def getLabels(path):
    return np.genfromtxt(path, dtype=int, delimiter=',', skip_header=0, encoding=None, usecols = (7))
    
def getTimes(path):
    timevector = np.genfromtxt(path, dtype=None, delimiter=',', skip_header=0, encoding=None, usecols=(1))
    print('!!!Time conversion happening!!!')
    epochvector = []
    for val in timevector:
        try:
            time = int(val[12:-7])
        except ValueError:
            time = int(val[11:-6])
       # print(time)
        epochvector.append(time)
    numpyepochs = np.array(epochvector)    
    return numpyepochs

def normalize(x):

    return x / x.max(axis=0)

#------------------------------KERAS PART

traindata = loadData(trainDataPath)
testdata = loadData(testDataPath)
testdata2 = loadData(testData2Path)
trainlabels = getLabels(trainDataPath)
testlabels = getLabels(testDataPath)
testlabels2 = getLabels(testData2Path)


print(traindata[1],traindata[2], traindata[3], traindata[4],traindata[5],traindata[6])


inputs = Input(shape=(5,))
x = Dense(8, activation='tanh')(inputs)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=optimizers.adam(lr=0.002),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.fit(traindata, trainlabels, validation_data=(testdata, testlabels), epochs=10, batch_size=1)

print('TEST DATA 1', model.evaluate(testdata, testlabels, batch_size=16))
print('TEST DATA 2', model.evaluate(testdata2, testlabels2, batch_size=16))


