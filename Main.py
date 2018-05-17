from DCGAN import *
from mnist import MNIST
import numpy as np


def main(gz=True):
    """
    @param: gz boolean, equals True if data is .gz (compressed) format. False otherwise
    """
    path_to_dataset = os.path.join('Datasets', 'MNIST')

    mndata = MNIST(path_to_dataset)
    mndata.gz = gz  # Données en format .gz dans le dossier Datasets\MNIST
    images, _ = mndata.load_training()
    images = np.array(images)/255
    dcgan = DCGAN([28, 28, 1], first_block_depth=20, model="intermediate")
    dcgan.train(images, 10, 100, k=1)#, strategy="min_gap")


if __name__ == '__main__':
    main()
