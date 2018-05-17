from DCGAN import *
from mnist import MNIST
import numpy as np
from os import listdir
from os.path import join
import pickle
import matplotlib.pyplot as plt


def main(dataSet='MNIST', model="simple"):
    """
    @param: gz boolean, for MNIST dataSete equals True if data is .gz (compressed) format. False otherwise
    @param: dataSet string, CIFAR MNIST
    """
    if dataSet == 'MNIST':
        path_to_dataset = os.path.join('Datasets', dataSet)

        mndata = MNIST(path_to_dataset)
        mndata.gz = True  # Données en format .gz dans le dossier Datasets\MNIST
        images, _ = mndata.load_training()
        images = np.array(images)
        dcgan = DCGAN([28, 28, 1], dim_noise=100, model=model, data=dataSet)
        dcgan.train(images, 50, 100, k=1)

    elif dataSet == 'CIFAR10':
        path_to_dataset = os.path.join('Datasets', dataSet)

        paths_to_batch = [os.path.join(path_to_dataset, f)for f in listdir(path_to_dataset) if f[0:10] == 'data_batch']

        with open(paths_to_batch[0], 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            labels = np.array(data[b'labels'])
            images = np.array(data[b'data'][np.where(labels == 7)])          

        for i in range(1,len(paths_to_batch)):
            with open(paths_to_batch[i], 'rb') as file:
                data = pickle.load(file, encoding='bytes')
                labels = np.array(data[b'labels'])
                images = np.append(images, np.array(data[b'data'][np.where(labels == 7)]),axis=0)
                
        images = images/255

        dcgan = DCGAN([32, 32, 3], dim_noise=300, model=model, data=dataSet)
        dcgan.train(images, 300, 100, k=5)
       
    elif dataSet == 'CelebA':
        images = np.load('CelebA_img.npz')['images']
        images = np.array(images)/255
        dcgan = DCGAN([218, 178, 3], dim_noise=400, model=model, data=dataSet)
        dcgan.train(images, 300, 100, k=1)


if __name__ == '__main__':
    main(model="dcgan")
