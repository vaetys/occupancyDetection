import os,sys
import random
import numpy as np
import tensorflow as tf
import filecmp
from keras.models import Model
from keras.layers import Input, Dense

seed = 7
np.random.seed(seed)

#----------------------------------------------------------------
trainDataPath = '../data/occupancy_data/datatraining.txt'
testDataPath = '../data/occupancy_data/datatest.txt'
#----------------------------------------------------------------

def loadData(path):
    data = np.genfromtxt(path, dtype=float, delimiter=',', skip_header=1, encoding=None, usecols = (2,3,4,5,6))
    return data

#Returns a spearate vector out of the last column of the given 2 dimensional numpy array.
def getLabels(path):
    return np.genfromtxt(path, dtype=int, delimiter=',', skip_header=1, encoding=None, usecols = (7))
    

#------------------------------KERAS PART

traindata = loadData('../data/occupancy_data/datatraining.txt')
testdata = loadData('../data/occupancy_data/datatest.txt')
trainlabels = getLabels(trainDataPath)
testlabels = getLabels(testDataPath)




inputs = Input(shape=(5,))
x = Dense(10, activation='relu')(inputs)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(traindata, trainlabels, validation_data=(testdata, testlabels), epochs=25)




