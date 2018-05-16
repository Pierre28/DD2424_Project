from DCGAN import *
from mnist import MNIST
import numpy as np


def main():
    path_to_dataset = os.path.join('Datasets', 'MNIST')

    mndata = MNIST(path_to_dataset)
    mndata.gz = True # Sous quel format sont vos données ? Moi ça marche quand je ne les décompresse pas mais je dois ajouter cette ligne
    # pour etre sur d'etre clair j'ai mes données en format .gz dans le dossier Datasets\MNIST
    images, _ = mndata.load_training()
    images = np.array(images)/255
    dcgan = DCGAN([28, 28, 1], first_block_depth=20, simple_model=True)
    dcgan.train(images, 100, 100)


if __name__ == '__main__':
    main()
