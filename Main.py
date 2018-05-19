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
        dcgan.train(images, 20, 64, k=2, is_inception_score_computed=False, is_model_saved=False, noise_type="gaussian")

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
                
        images = images

        dcgan = DCGAN([32, 32, 3], dim_noise=100, model=model, data=dataSet, flip_discri_labels=True)
        dcgan.train(images, 100, 64, k=1)
       
    elif dataSet == 'CelebA':
        images = np.load(os.path.join('Datasets','CelebA_img.npz'))['images']
        images = np.array(images)
        dcgan = DCGAN([218, 178, 3], dim_noise=400, model=model, data=dataSet)
        dcgan.train(images, 1, 5, k=1)


if __name__ == '__main__':
    main(dataSet='CIFAR10', model="intermediate")
