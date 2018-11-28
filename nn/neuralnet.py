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
#----------------------------------------------------------------

def loadData(path):
    data = np.genfromtxt(path, dtype=float, delimiter=',', skip_header=0, encoding=None, usecols = (1, 4))
    data[:,0] = getTimes(path)
    return data

#Returns a spearate vector out of the last column of the given 2 dimensional numpy array.
def getLabels(path):
    return np.genfromtxt(path, dtype=int, delimiter=',', skip_header=0, encoding=None, usecols = (7))
    
def getTimes(path):
    timevector = np.genfromtxt(path, dtype=None, delimiter=',', skip_header=0, encoding=None, usecols=(1))
    epochvector = []
    for val in timevector:
        time = int(val[12:-7])
        print(time)
        epochvector.append(time)
    numpyepochs = np.array(epochvector)    
    return numpyepochs



#------------------------------KERAS PART

traindata = loadData(trainDataPath)
testdata = loadData(testDataPath)
trainlabels = getLabels(trainDataPath)
testlabels = getLabels(testDataPath)

print(traindata[1], traindata[500])




inputs = Input(shape=(2,))
x = Dense(8, activation='tanh')(inputs)
y = Dense(20, activation='tanh')(x)
outputs = Dense(1, activation='sigmoid')(y)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model.fit(traindata, trainlabels, validation_data=(testdata, testlabels), epochs=10, batch_size=1)

print(model.evaluate(testdata, testlabels, batch_size=128))


