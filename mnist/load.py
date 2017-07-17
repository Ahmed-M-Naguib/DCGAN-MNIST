import sys
sys.path.append('..')

import numpy as np
import os

data_dir = 'data/'
def mnist():
    fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28*28)).astype(float)

    fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    return trX, teX, trY, teY

def mnist_with_valid_set():
    trX, teX, trY, teY = mnist()

    # traningSamplesPerClass = 500
    # testingSamplesPerClass = 100
    # numberOfClasses = 10
    #
    # indicesTrainClasses = []
    # indicesTestClasses = []
    # for i in np.arange(numberOfClasses):
    #     indicesTrainClasses = np.append(indicesTrainClasses, i*6000 + np.arange(traningSamplesPerClass))
    #     indicesTestClasses = np.append(indicesTestClasses, i * 1000 + np.arange(testingSamplesPerClass))
    #
    # indicesTrainClasses = indicesTrainClasses.astype(np.int64)
    # indicesTestClasses = indicesTestClasses.astype(np.int64)
    #
    #
    # trX = trX[indicesTrainClasses]
    # trY = trY[indicesTrainClasses]
    # teX = teX[indicesTestClasses]
    # teY = teY[indicesTestClasses]

    train_inds = np.arange(len(trX))
    np.random.shuffle(train_inds)
    trX = trX[train_inds]
    trY = trY[train_inds]
    #trX, trY = shuffle(trX, trY)
    vaX = trX[50000:]
    vaY = trY[50000:]
    trX = trX[:50000]
    trY = trY[:50000]

    return trX, vaX, teX, trY, vaY, teY