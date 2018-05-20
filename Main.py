from DCGAN import *
from mnist import MNIST
import numpy as np
from os import listdir
from os.path import join
import pickle
import matplotlib.pyplot as plt


def main(dataSet='MNIST', model="simple", dim_noise=100, flip_discri_labels=False,
         final_generator_activation="tanh", n_epochs=30, batch_size=100, k=1, is_data_normalized=False,
         is_inception_score_computed=False, is_model_saved=False, noise_type="uniform", strategy="k_steps"):
    """
    @param: gz boolean, for MNIST dataSete equals True if data is .gz (compressed) format. False otherwise
    @param: dataSet string, CIFAR MNIST
    """
    if dataSet == 'MNIST':
        path_to_dataset = os.path.join('Datasets', dataSet)

        mndata = MNIST(path_to_dataset)
        #mndata.gz = True  # Données en format .gz dans le dossier Datasets\MNIST
        images, _ = mndata.load_training()
        images = np.array(images)
        image_dimensions = [28, 28, 1]

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
        image_dimensions = [32, 32, 3]

    elif dataSet == 'CelebA':
        images = np.load(os.path.join('Datasets','CelebA_img.npz'))['images']
        images = np.array(images)
        image_dimensions = [218, 178, 3]

    dcgan = DCGAN(image_dimensions, dim_noise=dim_noise, model=model, data=dataSet,
                  flip_discri_labels=flip_discri_labels, final_generator_activation=final_generator_activation)
    dcgan.train(images, n_epochs, batch_size, k=k, is_inception_score_computed=is_inception_score_computed,
                is_model_saved=is_model_saved, noise_type=noise_type, is_data_normalized=is_data_normalized, strategy=strategy)


if __name__ == '__main__':
    main(dataSet='CIFAR10', model="dcgan", dim_noise=100, flip_discri_labels=False,
         final_generator_activation="tanh", n_epochs=100, batch_size=32, k=1, is_data_normalized=False,
         is_inception_score_computed=False, is_model_saved=False, noise_type="gaussian", strategy="probabilities")
