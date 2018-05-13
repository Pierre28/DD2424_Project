from DCGAN import *
from mnist import MNIST
import numpy as np
import tensorflow as tf


def main():
    mndata = MNIST('Datasets\MNIST')
    images, labels = mndata.load_training()
    images, labels = np.array(images), np.array(labels)
    dcgan = DCGAN([28, 28, 1])
    dcgan.train(np.array(images)[:10000, :], 100, 100)

if __name__ == '__main__':
    main()