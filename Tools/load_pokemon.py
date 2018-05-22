import numpy as np 
import cv2 
import os


def Import_pokemon(path_to_dataset):
    images = []
    count = 0
    images_repository = os.path.join('.', path_to_dataset, 'images')
    print(len(os.listdir(images_repository)))
    for img_path in os.listdir(images_repository):
        img = cv2.imread(os.path.join(images_repository, img_path))
        img = cv2.resize(img, (64, 64))
        images.append(np.array(img))
        count += 1
        if count%1000 == 0:
            print(count/100, '%')

    print(count, 'Pokemon images loaded')
    np.savez(os.path.join('.', path_to_dataset, 'pokemon.npz'), 'pokemon.npz', images=images)