from DCGAN import *
from mnist import MNIST
import numpy as np
import os
from Tools.load_pokemon import Import_pokemon
import pickle
import matplotlib.pyplot as plt


def main(dataSet='MNIST', test_name = '_1', model="simple", dim_noise=100, flip_discri_labels=False,
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
        images = np.array(images).reshape(-1, 28, 28, 1)
        image_dimensions = [28, 28, 1]

    elif dataSet == 'CIFAR10':
        path_to_dataset = os.path.join('Datasets', dataSet)
        paths_to_batch = [os.path.join(path_to_dataset, f)for f in os.listdir(path_to_dataset) if f[0:10] == 'data_batch']

        with open(paths_to_batch[0], 'rb') as file:
            data = pickle.load(file, encoding='bytes')
            labels = np.array(data[b'labels'])
            images = np.array(data[b'data'][np.where(labels == 7)])          

        for i in range(1,len(paths_to_batch)):
            with open(paths_to_batch[i], 'rb') as file:
                data = pickle.load(file, encoding='bytes')
                labels = np.array(data[b'labels'])
                images = np.append(images, np.array(data[b'data'][np.where(labels == 7)]),axis=0)
        images = np.reshape(images, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
        image_dimensions = [32, 32, 3]

    elif dataSet == 'CelebA':
        images = np.load(os.path.join('Datasets','CelebA_img.npz'))['images']
        images = np.array(images)
        image_dimensions = [218, 178, 3]

    elif dataSet == 'pokemon':
        path_to_dataset_directory = os.path.join('Datasets', dataSet)
        path_to_dataset_file = os.path.join(path_to_dataset_directory, 'pokemon.npz')
        if not os.path.exists(path_to_dataset_file):
            Import_pokemon(path_to_dataset_directory)
        with np.load(path_to_dataset_file) as data:
            images = data['images']
        image_dimensions = images[0].shape

    dcgan = DCGAN(image_dimensions, dim_noise=dim_noise, model=model, data=dataSet,
                  flip_discri_labels=flip_discri_labels, final_generator_activation=final_generator_activation, test_name = test_name)
    dcgan.train(images, n_epochs, batch_size, k=k, is_inception_score_computed=is_inception_score_computed,
                is_model_saved=is_model_saved, noise_type=noise_type, is_data_normalized=is_data_normalized, strategy=strategy)


if __name__ == '__main__':
    main(dataSet='CIFAR10', test_name = '_1', model="intermediate", dim_noise=100, flip_discri_labels=False,
         final_generator_activation="tanh", n_epochs=150, batch_size=32, k=1, is_data_normalized=False,
         is_inception_score_computed=False, is_model_saved=False, noise_type="gaussian", strategy="k_steps")
