import inception_model
from mnist import MNIST
import numpy as np        
from six.moves import cPickle 


def LoadBatch(filename):
	f = open(filename, 'rb')
	datadict = cPickle.load(f,encoding='latin1')
	f.close()

	X = np.array(datadict["data"], dtype = 'float64')
	y = datadict['labels']
	y = np.array(y)

	return X.transpose(), y  #shape(X.transpose) : dxN, size(Y) : kxN, size(y) = N

def main():
    ### MNIST
    # mndata = MNIST('Datasets/MNIST')
    # images, labels = mndata.load_training()

    ###### CIFAR
    images_, labels = LoadBatch('../DD2424/Datasets/cifar-10-batches-py/data_batch_1')
    images = []
    for image in range(images_.shape[1]):
    	images.append(images_[:,image].reshape((32,32,3)))





    #Images doit être une liste, avec chaque element représentant une image, type = np array
    ###### Inception score
    mean, std = inception_model.get_inception_score(images[:1000])
    print(mean, std)

if __name__ == '__main__':
    main()