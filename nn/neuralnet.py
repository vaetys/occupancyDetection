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
from cloudant import couchdb
import json
import dbconnection as db

#converts a given json dictinoary to a matrix. Json dict must contain a json-
#array called 'docs'
def jsonToMatrix(docs):
    matrix = []
    for doc in docs['docs']:
        row = []
        for key in doc:
            row.append(doc[key])
        matrix.append(row)
    return matrix


#Loads the data from a text file and returns a ready-to-use numpy-array
def loadData(dbname):
    arrayData = jsonToMatrix(db.fetchAll(dbname))
    data = np.array(arrayData)
    data = np.array(data[:, [4,5,6,7,8]], dtype=float) #the chosen indices determine what features to use
    np.append(getTimes(dbname), data, axis=1) #
    data = normalize(data)
    return data


#Returns a spearate vector out of the last column of the given 2 dimensional numpy array.
def getLabels(dbname):
    labelData = db.fetchLabels(dbname) 
    data = np.array(labelData)
    return data

#returns the hour-value for the dates in a given database name.
def getTimes(dbname):
    timevector = db.fetchTimes(dbname)
    a = len(timevector)
    print('!!!Time conversion happening!!!')
    hourvector = []
    for val in timevector:
        try:
            time = int(val[12:-7])
        except ValueError:
            time = int(val[11:-6])
        hourvector.append(time)
    numpyepochs = np.array(hourvector)    
    return numpyepochs.reshape(a, 1)


#Returns the normalized version of data
def normalize(x):
    return x / x.max(axis=0)

traindata = loadData('occupancytraining')
testdata = loadData('occupancytest1')
testdata2 = loadData('occupancytest2')
trainlabels = getLabels('occupancytraining')
testlabels = getLabels('occupancytest1')
testlabels2 = getLabels('occupancytest2')

inputs = Input(shape=(5,))
x = Dense(8, activation='tanh')(inputs)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=optimizers.adam(lr=0.002),
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(traindata, trainlabels, validation_data=(testdata, testlabels), epochs=10, batch_size=32)

print('TEST DATA 1', model.evaluate(testdata, testlabels, batch_size=16))
print('TEST DATA 2', model.evaluate(testdata2, testlabels2, batch_size=16))


