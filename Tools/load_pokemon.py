import numpy as np 
import cv2 
import os

def Import_pokemon(path_to_dataset):
	images = []
	count = 0
	for img_path in os.listdir(path_to_dataset)[1:]:
		img = cv2.imread(os.path.join(path_to_dataset, img_path))
		img = cv2.resize(img, (64, 64))
		images.append(np.array(img))
		print(np.array(img).shape)
		count += 1
		if count%1000 == 0:
			print(count/100, '%')
			
	print(count, 'Pokemon images loaded')
	np.savez('Datasets/pokemon.npz', images = images)




Import_pokemon(os.path.join('Datasets','pokemon'))

