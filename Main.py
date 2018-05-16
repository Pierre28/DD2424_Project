from DCGAN import *
from mnist import MNIST
import numpy as np


def main():
    path_to_dataset = os.path.join('Datasets', 'MNIST')

    mndata = MNIST(path_to_dataset)
    images, _ = mndata.load_training()
    images = np.array(images)/255
    dcgan = DCGAN([28, 28, 1], first_block_depth=20, simple_model=True)
    dcgan.train(images, 1000, 100)


if __name__ == '__main__':
    main()
