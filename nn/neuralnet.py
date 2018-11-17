import os,sys
import random
import numpy as np
import tensorflow as tf
import filecmp


#----------------------------------------------------------------
trainDataPath = '../data/occupancy_data/datatraining.txt'
# testDataPath = os.path.join(ROOT_PATH, "data/occupancy_data/datatest.txt")
#----------------------------------------------------------------

def loadData(path):
    data = np.genfromtxt(path, dtype=None, delimiter=',', names=True, skip_header=2, encoding='ASCII')
    return data

def printData():
    trainData = loadData(trainDataPath)
    print(trainData[1])

printData()


