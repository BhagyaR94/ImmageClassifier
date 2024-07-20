import pickle

import cv2
from skimage.transform import resize
import numpy as np
from skimage.io import imread

print('helloooo')

EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("model.p", "rb"))

sample_img = imread('./car.jpg')
empty_slot = imread('./00000000_00000161.jpg')

cv2.imshow('img', sample_img)
cv2.imshow('img2', empty_slot)
cv2.waitKey(0)


path_arr = [sample_img, empty_slot]

for path in path_arr:

    flat_data = []
    img_resized = resize(path, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)
    print('', y_output)

    if y_output == 0:
        print('Empty: ', EMPTY)
    else:
        print('Empty', NOT_EMPTY)
