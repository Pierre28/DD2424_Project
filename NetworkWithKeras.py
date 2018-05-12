import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from six.moves import cPickle
import numpy as np

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def loadBatch(file):
    """Extraction of data from files
    @:return X.T=Set of images easily displayable (as ndarray (N, d)), X/255=normalized images (d, N),
    Y=one-hot encoded labels (K, N), y=original labels (list of len N)"""
    f = open(file, 'rb')
    datadict = cPickle.load(f,encoding='latin1') # keys : ['batch_label', 'labels', 'data', 'filenames']
    f.close()
    X = datadict['data']
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    X = X.reshape(X.shape[0], 32, 32, 3)  # ndarray of shape (N, d=3072)
    y = datadict['labels']  # list of len N
    Y = keras.utils.to_categorical(y, num_classes=10)
    return X, Y

def loadAugmentedBatches(files_without_last_character, val_size):
    """Get bigger training set and a validation set of val_size images.
    @:return images_train, X_train, Y_train, y_train, images_val, X_val, Y_val, y_val"""
    X, Y = loadBatch(files_without_last_character + str(1))
    for i in range(2, 6):
        X_to_add, Y_to_add = loadBatch(files_without_last_character + str(i))
        X = np.concatenate((X, X_to_add))
        Y = np.concatenate((Y, Y_to_add))
    return X[:-val_size, :], Y[:-val_size, :], X[-val_size:, :], Y[-val_size:, :]

def normalize_data(X_train, X_test):
    """Normalize the input data: substract the mean and divide by the standard deviation, calculated from the training
    set only.
    @:return Normalized X_train, X_val, X_test"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean)/std
    X_test = (X_test - mean) / std
    return X_train, X_test

# Get data
#x_train, y_train, x_val, y_val = loadAugmentedBatches('Datasets/cifar-10-batches-py/data_batch_', 1000)
x_train, y_train = loadBatch('Datasets/cifar-10-batches-py/data_batch_1')
x_val, y_val = loadBatch('Datasets/cifar-10-batches-py/data_batch_1')
x_test, y_test = loadBatch('Datasets/cifar-10-batches-py/test_batch')
#x_train, x_test = normalize_data(x_train, x_test)
#x_train = keras.utils.normalize(x_train, axis=-1, order=2)
#x_test = keras.utils.normalize(x_train, axis=-1, order=2)

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


model.fit(x_train, y_train,
          epochs=10,
          batch_size=32, validation_data=(x_val, y_val))
score = model.evaluate(x_test, y_test, batch_size=32)
print(score)
