import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def img_resize(image, y_dim, x_dim):
    resized_img = cv2.resize(image, (y_dim, x_dim))
    return resized_img

def normalize_array(arr):
    return arr / 255.0

import glob

def load_data(data_dir, image_size, test_size=0.2, random_state=None):
    # Define paths
    image_path = os.path.join(data_dir, "images", "*.png")
    mask_path = os.path.join(data_dir, "masks", "*.png")

    image_names = sorted(glob.glob(image_path), key=lambda x: x.split('.')[0])
    mask_names = sorted(glob.glob(mask_path), key=lambda x: x.split('.')[0])

    train_images_array = []

    for image in image_names:
        img = cv2.imread(image, -1)
        img = img_resize(img, image_size, image_size)
        train_images_array.append(img)

    train_images_array = np.array(train_images_array)

    mask_images_array = []

    for mask in mask_names:
        msk = cv2.imread(mask, -1)
        msk = img_resize(msk, image_size, image_size)
        mask_images_array.append(msk)

    mask_images_array = np.array(mask_images_array)

    images = normalize_array(train_images_array)
    masks = normalize_array(mask_images_array)

    X_train, X_test, y_train, y_test = train_test_split(
        images, masks, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
